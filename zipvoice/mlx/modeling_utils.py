import json
from typing import Optional

import numpy as np

import librosa
import mlx.core as mx

import torch
from transformers import pipeline
from huggingface_hub import snapshot_download
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

from torch.nn.utils import parametrize
from linacodec.vocoder.vocos import Vocos

def _synth_prompt(duration: float, sample_rate: int) -> np.ndarray:
    length = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, length, endpoint=False, dtype=np.float32)
    return 0.01 * np.sin(2.0 * np.pi * 220.0 * t)


def _apply_fade_np(wav: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    fade_samples = max(0, int(sample_rate * max(fade_ms, 0.0) / 1000.0))
    if fade_samples <= 1:
        return wav
    if wav.shape[-1] <= fade_samples * 2:
        return wav
    out = wav.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out

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


def load_transcriber_mlx():
    torch_device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch_device == "mps" else -1,
    )


def load_models_mlx(
    model_path: Optional[str] = None,
    device: str = "cpu",
    vocoder_backend: str = "mlx",
    vocoder_device: Optional[str] = None,
    load_transcriber: bool = False,
):
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    model_path = str(model_path)
    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    transcriber = load_transcriber_mlx() if load_transcriber else None

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

    feature_extractor = None
    if vocoder_backend == "torch":
        if vocoder_device is None:
            vocoder_device = "mps" if torch.backends.mps.is_available() else "cpu"
        vocos = load_vocoder_torch(model_path, device=vocoder_device)
    else:
        vocos = load_vocoder_mlx(model_path)

    return model, feature_extractor, vocos, tokenizer, transcriber


def _compute_num_frames(num_samples: int, hop_length: int) -> int:
    return int((int(num_samples) + int(hop_length) // 2) // int(hop_length))


def _rms_norm_np(prompt_wav: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    wav = prompt_wav.astype(np.float32, copy=False)
    prompt_rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64) + 1.0e-12))
    if prompt_rms < target_rms and prompt_rms > 0:
        wav = wav * (target_rms / prompt_rms)
    return wav.astype(np.float32, copy=False), prompt_rms


def _extract_vocos_fbank_np(
    samples: np.ndarray,
    sampling_rate: int = 24000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32, copy=False),
        sr=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=1.0,
    )
    logmel = np.log(np.clip(mel, 1.0e-7, None)).T.astype(np.float32, copy=False)

    num_frames = _compute_num_frames(samples.shape[-1], hop_length)
    if logmel.shape[0] > num_frames:
        logmel = logmel[:num_frames]
    elif logmel.shape[0] < num_frames:
        if logmel.shape[0] == 0:
            logmel = np.zeros((num_frames, n_mels), dtype=np.float32)
        else:
            pad = num_frames - logmel.shape[0]
            logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="edge")
    return logmel


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
    offset: float = 0.0,
    fade_ms: float = 12.0,
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
        prompt_wav, _sr = librosa.load(audio, sr=24000, offset=max(offset, 0.0), duration=duration)
        if not prompt_text:
            prompt_wav2, _sr2 = librosa.load(audio, sr=16000, offset=max(offset, 0.0), duration=duration)
            if transcriber is None:
                raise RuntimeError(
                    "Prompt transcription is unavailable. Pass --prompt-text or provide a transcriber."
                )
            prompt_text = transcriber(prompt_wav2)["text"]
            print(prompt_text)

    # Reduce click/noise at prompt boundaries and remove DC bias.
    prompt_wav = _apply_fade_np(prompt_wav, 24000, fade_ms=fade_ms)
    prompt_wav = prompt_wav - float(np.mean(prompt_wav))

    prompt_wav, prompt_rms = _rms_norm_np(prompt_wav, target_rms)
    prompt_features = _extract_vocos_fbank_np(prompt_wav, sampling_rate=24000)
    prompt_features = np.expand_dims(prompt_features, axis=0) * float(feat_scale)
    prompt_features_lens = np.array([prompt_features.shape[1]], dtype=np.int64)

    if not prompt_text:
        prompt_text = "Hello."
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features.astype(np.float32), float(prompt_rms)


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
    speed=0.92,
    t_shift=0.5,
    target_rms=0.01,
    duration_pad_frames=16,
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
