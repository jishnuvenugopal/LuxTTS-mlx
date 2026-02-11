import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import librosa
import torchaudio
from transformers import pipeline
from huggingface_hub import snapshot_download
from lhotse.utils import fix_random_seed

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

from dataclasses import dataclass, field
from typing import Optional, List

from linacodec.vocoder.vocos import Vocos
from zipvoice.onnx_modeling import OnnxModel
from torch.nn.utils import parametrize
from zipvoice.vocoder_patches import apply_linacodec_linkwitz_patch


@dataclass
class LuxTTSConfig:
    # Model Setup
    model_dir: Optional[str] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[str] = None
    trt_engine_path: Optional[str] = None

    # Tokenizer & Language
    tokenizer: str = "emilia"  # choices: ["emilia", "libritts", "espeak", "simple"]
    lang: str = "en-us"


@torch.inference_mode
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


def _trim_silence_edges_np(
    wav: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    keep_silence_ms: float,
) -> np.ndarray:
    if wav.size == 0:
        return wav

    abs_wav = np.abs(wav.astype(np.float32, copy=False))
    peak = float(np.max(abs_wav))
    if peak <= 1.0e-8:
        return wav

    threshold = peak * float(10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(abs_wav > threshold)
    if active.size == 0:
        return wav

    keep = max(0, int(sample_rate * max(keep_silence_ms, 0.0) / 1000.0))
    start = max(0, int(active[0]) - keep)
    end = min(wav.shape[-1], int(active[-1]) + keep + 1)
    if end - start <= 8:
        return wav
    return wav[start:end]


def _normalize_prompt_rms_torch(
    prompt_wav: torch.Tensor,
    target_rms: float,
    rms_min: float,
    rms_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    min_rms = max(0.0, float(rms_min))
    max_rms = max(min_rms + 1.0e-6, float(rms_max))
    safe_target = float(np.clip(float(target_rms), min_rms, max_rms))

    prompt_wav, prompt_rms = rms_norm(prompt_wav, safe_target)
    prompt_rms_value = float(prompt_rms.detach().cpu().item())
    if prompt_rms_value > max_rms and prompt_rms_value > 1.0e-8:
        prompt_wav = prompt_wav * (max_rms / prompt_rms_value)
    return prompt_wav, prompt_rms


def process_audio(
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
    trim_silence: bool = True,
    silence_threshold_db: float = -42.0,
    keep_silence_ms: float = 35.0,
    rms_min: float = 0.006,
    rms_max: float = 0.03,
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
        prompt_wav, sr = librosa.load(audio, sr=24000, offset=max(offset, 0.0), duration=duration)
        if not prompt_text:
            if transcriber is None:
                raise RuntimeError(
                    "Prompt transcription is unavailable. Pass --prompt-text or install/cache a Whisper model."
                )
            prompt_wav2, sr = librosa.load(audio, sr=16000, offset=max(offset, 0.0), duration=duration)
            prompt_text = transcriber(prompt_wav2)["text"]
            print(prompt_text)

    if trim_silence:
        prompt_wav = _trim_silence_edges_np(
            prompt_wav,
            sample_rate=24000,
            threshold_db=silence_threshold_db,
            keep_silence_ms=keep_silence_ms,
        )

    prompt_wav = _apply_fade_np(prompt_wav, 24000, fade_ms=fade_ms)
    prompt_wav = prompt_wav - float(np.mean(prompt_wav))

    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = _normalize_prompt_rms_torch(
        prompt_wav,
        target_rms=target_rms,
        rms_min=rms_min,
        rms_max=rms_max,
    )

    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=24000
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    if not prompt_text:
        prompt_text = "Hello."
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms

def generate(
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
    device = next(model.parameters()).device  # Auto-detect device

    with torch.inference_mode():
        (pred_features, _, _, _) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=speed,
            duration_pad_frames=duration_pad_frames,
            t_shift=t_shift,
            duration='predict',
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / 0.1
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)

    return wav

def load_models_gpu(model_path=None, device="cuda"):
    params = LuxTTSConfig()
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = ZipVoiceDistill(
        **model_config["model"],
        **tokenizer_config,
    )
    load_checkpoint(filename=model_ckpt, model=model, strict=True)
    params.device = torch.device(device, 0)

    model = model.to(params.device).eval()
    feature_extractor = VocosFbank()

    apply_linacodec_linkwitz_patch()
    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').to(device)
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=params.device))

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    return model, feature_extractor, vocos, tokenizer, transcriber

def load_models_cpu(model_path = None, num_thread=2):
    params = LuxTTSConfig()
    params.seed = 42

    model_path = snapshot_download('YatharthS/LuxTTS')

    token_file = f"{model_path}/tokens.txt"
    text_encoder_path = f"{model_path}/text_encoder.onnx"
    fm_decoder_path = f"{model_path}/fm_decoder.onnx"
    model_config  = f"{model_path}/config.json"

    transcriber = None
    for asr_model_id in ("openai/whisper-tiny", "openai/whisper-base"):
        try:
            asr_model_path = snapshot_download(asr_model_id, local_files_only=True)
        except Exception:
            continue
        try:
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=asr_model_path,
                device="cpu",
            )
            break
        except Exception:
            continue

    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)

    apply_linacodec_linkwitz_patch()
    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').eval()
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=torch.device('cpu')))

    feature_extractor = VocosFbank()

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    params.onnx_int8 = True
    return model, feature_extractor, vocos, tokenizer, transcriber
