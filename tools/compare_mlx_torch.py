#!/usr/bin/env python3
import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import mlx.core as mx

from huggingface_hub import snapshot_download

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.feature import VocosFbank

from zipvoice.mlx.model import ZipVoiceDistillMLX
from zipvoice.mlx.modeling_utils import process_audio_mlx
from zipvoice.mlx.weights import apply_state_dict_mlx
from zipvoice.modeling_utils import process_audio


def _to_np(x):
    if isinstance(x, mx.array):
        return np.array(x)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def _stats(name, a, b):
    a_np = _to_np(a)
    b_np = _to_np(b)
    if a_np.shape != b_np.shape:
        print(f"{name}: shape mismatch {a_np.shape} vs {b_np.shape}")
        return
    diff = a_np - b_np
    print(
        f"{name}: mean_abs={np.mean(np.abs(diff)):.6e} "
        f"max_abs={np.max(np.abs(diff)):.6e} "
        f"a_mean={np.mean(a_np):.6e} b_mean={np.mean(b_np):.6e}"
    )


def _load_torch_state_dict(model_ckpt: str):
    checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _load_torch_model(model_path: str, tokenizer: EmiliaTokenizer, device: str) -> ZipVoiceDistill:
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"
    with open(model_config, "r") as f:
        config = json.load(f)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    model = ZipVoiceDistill(**config["model"], **tokenizer_config)
    load_checkpoint(filename=model_ckpt, model=model, strict=True)
    model = model.to(device).eval()
    return model


def _load_mlx_model(model_path: str, tokenizer: EmiliaTokenizer) -> ZipVoiceDistillMLX:
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"
    with open(model_config, "r") as f:
        config = json.load(f)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    model = ZipVoiceDistillMLX(**config["model"], **tokenizer_config)
    state_dict = _load_torch_state_dict(model_ckpt)
    apply_state_dict_mlx(model, state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare MLX and Torch ZipVoice outputs.")
    parser.add_argument("--model-path", default=None, help="Local model path (default: HF YatharthS/LuxTTS)")
    parser.add_argument("--prompt", required=True, help="Path to prompt wav/mp3.")
    parser.add_argument("--prompt-text", default=None, help="Prompt transcript (skip Whisper).")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for x0.")
    parser.add_argument("--device", default="cpu", help="Torch device for comparison (cpu or mps).")
    args = parser.parse_args()

    if args.model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")
    else:
        model_path = args.model_path

    token_file = f"{model_path}/tokens.txt"
    tokenizer = EmiliaTokenizer(token_file=token_file)

    torch_model = _load_torch_model(model_path, tokenizer, args.device)
    mlx_model = _load_mlx_model(model_path, tokenizer)

    # Prompt features
    feature_extractor = VocosFbank()
    prompt_text = args.prompt_text or args.text
    prompt_tokens_t, prompt_features_lens_t, prompt_features_t, prompt_rms_t = process_audio(
        args.prompt,
        transcriber=None,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=args.device,
        target_rms=0.01,
        duration=5,
        prompt_text=prompt_text,
    )

    prompt_tokens_m, prompt_features_lens_m, prompt_features_m, prompt_rms_m = process_audio_mlx(
        args.prompt,
        transcriber=None,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=args.device,
        target_rms=0.01,
        duration=5,
        prompt_text=prompt_text,
    )

    # ensure torch tensors on correct device
    prompt_features_t = prompt_features_t.to(args.device)
    prompt_features_lens_t = prompt_features_lens_t.to(args.device)

    tokens = tokenizer.texts_to_token_ids([args.text])

    # Text encoder outputs
    torch_embed, torch_tokens_lens = torch_model.forward_text_embed(tokens)
    mlx_embed, mlx_tokens_lens = mlx_model.forward_text_embed(tokens)
    _stats("text_embed", torch_embed, mlx_embed)

    # Text condition and masks
    torch_text_cond, torch_pad = torch_model.forward_text_inference_ratio_duration(
        tokens=tokens,
        prompt_tokens=prompt_tokens_t,
        prompt_features_lens=prompt_features_lens_t,
        speed=1.0,
    )
    mlx_text_cond, mlx_pad = mlx_model.forward_text_inference_ratio_duration(
        tokens=tokens,
        prompt_tokens=prompt_tokens_m,
        prompt_features_lens=mx.array(prompt_features_lens_m.cpu().numpy(), dtype=mx.int64),
        speed=1.0,
    )
    _stats("text_condition", torch_text_cond, mlx_text_cond)

    # Speech condition for fm decoder
    num_frames = torch_text_cond.shape[1]
    torch_speech = torch.nn.functional.pad(
        prompt_features_t,
        (0, 0, 0, num_frames - prompt_features_t.shape[1]),
    )
    torch_mask = torch_pad.unsqueeze(-1)
    torch_speech = torch.where(torch_mask, torch.zeros_like(torch_speech), torch_speech)

    mlx_num_frames = int(mx_text_cond.shape[1])
    prompt_features_m_np = prompt_features_m.cpu().numpy()
    mlx_speech = mx.pad(
        mx.array(prompt_features_m_np),
        ((0, 0), (0, mlx_num_frames - prompt_features_m.shape[1]), (0, 0)),
    )
    mlx_mask = mx.expand_dims(mlx_pad, axis=-1)
    mlx_speech = mx.where(mlx_mask, mx.zeros_like(mlx_speech), mlx_speech)

    # Compare fm decoder output for a fixed x0 and t
    rng = np.random.default_rng(args.seed)
    x0_np = rng.standard_normal((1, num_frames, torch_text_cond.shape[-1])).astype(np.float32)

    t_torch = torch.tensor(0.5, dtype=torch.float32)
    vt_torch = torch_model.forward_fm_decoder(
        t=t_torch,
        xt=torch.from_numpy(x0_np),
        text_condition=torch_text_cond,
        speech_condition=torch_speech,
        padding_mask=torch_pad,
    )

    t_mlx = mx.array(0.5, dtype=mx.float32)
    vt_mlx = mlx_model.forward_fm_decoder(
        t=t_mlx,
        xt=mx.array(x0_np),
        text_condition=mlx_text_cond,
        speech_condition=mlx_speech,
        padding_mask=mlx_pad,
    )
    _stats("fm_decoder", vt_torch, vt_mlx)

    print("Done.")


if __name__ == "__main__":
    main()
