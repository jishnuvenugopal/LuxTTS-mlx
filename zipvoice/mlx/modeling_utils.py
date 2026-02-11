import json
from typing import Optional

import numpy as np

import librosa
import mlx.core as mx

import torch
from transformers import pipeline
from huggingface_hub import snapshot_download
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

from torch.nn.utils import parametrize
from linacodec.vocoder.vocos import Vocos

def _synth_prompt(duration: float, sample_rate: int) -> np.ndarray:
    length = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, length, endpoint=False, dtype=np.float32)
    return 0.01 * np.sin(2.0 * np.pi * 220.0 * t)

from .model import ZipVoiceDistillMLX
from .weights import apply_state_dict_mlx
from .vocoder import load_vocoder_mlx


def _load_torch_state_dict(model_ckpt: str):
    checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_vocoder_torch(model_path: str, device: str = "cpu"):
    vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml").to(device).eval()
    for layer in getattr(vocos.upsampler, "upsample_layers", []):
        try:
            parametrize.remove_parametrizations(layer, "weight")
        except ValueError:
            continue
    vocos.load_state_dict(torch.load(f"{model_path}/vocoder/vocos.bin", map_location=device))
    return vocos


def load_models_mlx(
    model_path: Optional[str] = None,
    device: str = "cpu",
    vocoder_backend: str = "mlx",
    vocoder_device: Optional[str] = None,
):
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    model_path = str(model_path)
    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    torch_device = "mps" if torch.backends.mps.is_available() else "cpu"
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch_device == "mps" else -1,
    )

    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = ZipVoiceDistillMLX(
        **model_config["model"],
        **tokenizer_config,
    )

    state_dict = _load_torch_state_dict(model_ckpt)
    apply_state_dict_mlx(model, state_dict)

    feature_extractor = VocosFbank()
    if vocoder_backend == "torch":
        if vocoder_device is None:
            vocoder_device = "mps" if torch.backends.mps.is_available() else "cpu"
        vocos = load_vocoder_torch(model_path, device=vocoder_device)
    else:
        vocos = load_vocoder_mlx(model_path)

    return model, feature_extractor, vocos, tokenizer, transcriber


@torch.inference_mode
def process_audio_mlx(
    audio,
    transcriber,
    tokenizer,
    feature_extractor,
    device,
    target_rms=0.1,
    duration=4,
    feat_scale=0.1,
    prompt_text: Optional[str] = None,
):
    if audio is None:
        prompt_wav = _synth_prompt(duration, 24000)
        if not prompt_text:
            prompt_text = "Hello."
    elif isinstance(audio, np.ndarray):
        prompt_wav = audio.astype(np.float32)
        if not prompt_text:
            prompt_text = "Hello."
    else:
        prompt_wav, _sr = librosa.load(audio, sr=24000, duration=duration)
        if not prompt_text:
            prompt_wav2, _sr2 = librosa.load(audio, sr=16000, duration=duration)
            prompt_text = transcriber(prompt_wav2)["text"]
            print(prompt_text)

    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=24000).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    if not prompt_text:
        prompt_text = "Hello."
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms


def _to_numpy(x):
    if isinstance(x, mx.array):
        return np.array(x)
    return np.array(x)


def generate_mlx(
    prompt_tokens,
    prompt_features_lens,
    prompt_features,
    prompt_rms,
    text,
    model,
    vocoder,
    tokenizer,
    num_step=4,
    guidance_scale=3.0,
    speed=1.0,
    t_shift=0.5,
    target_rms=0.1,
    duration_pad_frames=10,
):
    tokens = tokenizer.texts_to_token_ids([text])

    prompt_features = mx.array(prompt_features)
    prompt_features_lens = mx.array(prompt_features_lens, dtype=mx.int64)

    pred_features, _, _, _ = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        duration_pad_frames=duration_pad_frames,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    pred_features = pred_features / 0.1

    if isinstance(vocoder, torch.nn.Module):
        device = next(vocoder.parameters()).device
        pred_np = _to_numpy(mx.transpose(pred_features, (0, 2, 1)))
        pred_t = torch.from_numpy(pred_np).to(device)
        wav = vocoder.decode(pred_t).squeeze(1).clamp(-1, 1)
        wav = wav.cpu().numpy()
        wav = mx.array(wav)
    else:
        wav = vocoder.decode(pred_features)

    prompt_rms_value = float(prompt_rms)
    if prompt_rms_value < target_rms:
        wav = wav * (prompt_rms_value / target_rms)

    return wav
