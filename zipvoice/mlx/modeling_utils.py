import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import mlx.core as mx

import torch
from transformers import pipeline
from huggingface_hub import snapshot_download
from torch.nn.utils import parametrize

from linacodec.vocoder.vocos import Vocos
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.feature import VocosFbank

from .model import ZipVoiceDistillMLX


def _load_torch_state_dict(model_ckpt: str):
    checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _set_param(obj, name: str, value: np.ndarray):
    parts = name.split(".")
    cur = obj
    for part in parts[:-1]:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    leaf = parts[-1]

    # Conv1d weight in MLX expects (out, kernel, in)
    if leaf == "weight" and cur.__class__.__name__ == "Conv1d":
        if value.ndim == 3:
            value = np.transpose(value, (0, 2, 1))
    setattr(cur, leaf, mx.array(value))


def apply_state_dict_mlx(model, state_dict):
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().numpy()
        _set_param(model, name, value)


def _load_vocos_torch(model_path: str, device: str = "cpu"):
    vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml").eval()
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f"{model_path}/vocoder/vocos.bin", map_location=device))
    vocos = vocos.to(device)
    return vocos


def load_models_mlx(model_path: Optional[str] = None, device: str = "cpu"):
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
    vocos = _load_vocos_torch(model_path, device=torch_device)

    return model, feature_extractor, vocos, tokenizer, transcriber


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
):
    tokens = tokenizer.texts_to_token_ids([text])

    prompt_features = mx.array(prompt_features)
    prompt_features_lens = mx.array(prompt_features_lens, dtype=mx.int64)

    pred_features, _, _, _ = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed * 1.3,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    pred_features = mx.transpose(pred_features, (0, 2, 1)) / 0.1

    if isinstance(vocoder, torch.nn.Module):
        device = next(vocoder.parameters()).device
        pred_np = _to_numpy(pred_features)
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
