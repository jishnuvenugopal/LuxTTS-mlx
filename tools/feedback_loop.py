#!/usr/bin/env python3
"""
Self-tuning quality loop for LuxTTS MLX cloning.

It runs repeated synthesis rounds, compares available backends (MLX vocoder vs
torch vocoder), scores each run, and updates parameters until thresholds pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import pipeline


@dataclass
class PromptCase:
    path: Path
    prompt_text: str
    name: str


@dataclass
class SynthesisParams:
    num_steps: int = 5
    guidance_scale: float = 3.0
    t_shift: float = 0.5
    speed: float = 0.92
    duration_pad_frames: int = 16
    ref_duration: float = 2.4
    prompt_start: float = 0.0
    prompt_fade_ms: float = 12.0
    trim_silence: bool = True
    silence_threshold_db: float = -42.0
    keep_silence_ms: float = 35.0
    rms: float = 0.01
    rms_min: float = 0.006
    rms_max: float = 0.03
    return_smooth: bool = True


@dataclass
class QualityThresholds:
    min_asr_sim: float = 0.78
    max_start_ratio: float = 0.45
    max_tail_ratio: float = 0.58
    max_clip_frac: float = 0.0005
    max_repeat_ratio: float = 0.55


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav), dtype=np.float64) + 1.0e-12))


def _slice_safe(wav: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, wav.size))
    end = max(0, min(end, wav.size))
    if end <= start:
        return np.zeros((0,), dtype=np.float32)
    return wav[start:end]


def _repeat_ratio(text: str) -> float:
    norm = _normalize_text(text)
    tokens = norm.split()
    if not tokens:
        return 1.0
    if len(tokens) == 1:
        return 1.0

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    dominant = max(counts.values()) / float(len(tokens))

    repeats = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeats += 1
    adjacent = repeats / float(max(1, len(tokens) - 1))
    return float(max(dominant, adjacent))


def _score_run(metrics: dict[str, float | None]) -> float:
    asr = metrics.get("asr_sim")
    asr_term = 3.2 * float(asr if asr is not None else 0.70)
    repeat_ratio = float(metrics.get("repeat_ratio", 1.0) or 1.0)
    repeat_penalty = 1.8 * max(0.0, repeat_ratio - 0.35)
    low_asr_penalty = 0.9 * max(0.0, 0.30 - float(asr or 0.0))
    return float(
        asr_term
        - 0.7 * float(metrics["start_ratio"])
        - 0.8 * float(metrics["tail_ratio"])
        - 10.0 * float(metrics["clip_frac"])
        - repeat_penalty
        - low_asr_penalty
    )


def _passes_thresholds(metrics: dict[str, float | None], thresholds: QualityThresholds) -> bool:
    if metrics["clip_frac"] > thresholds.max_clip_frac:
        return False
    if metrics["start_ratio"] > thresholds.max_start_ratio:
        return False
    if metrics["tail_ratio"] > thresholds.max_tail_ratio:
        return False
    repeat_ratio = metrics.get("repeat_ratio")
    if repeat_ratio is not None and repeat_ratio > thresholds.max_repeat_ratio:
        return False
    asr_sim = metrics.get("asr_sim")
    if asr_sim is not None and asr_sim < thresholds.min_asr_sim:
        return False
    return True


def _load_local_asr():
    device = 0 if torch.backends.mps.is_available() else -1
    for model_id in ("openai/whisper-base", "openai/whisper-tiny"):
        try:
            model_path = snapshot_download(model_id, local_files_only=True)
        except Exception:
            continue
        try:
            return pipeline("automatic-speech-recognition", model=model_path, device=device)
        except Exception:
            continue
    return None


def _prompt_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / max(1, info.samplerate))


def _load_cases(args, asr_pipe) -> list[PromptCase]:
    cases: list[PromptCase] = []
    if args.manifest:
        with Path(args.manifest).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"prompt", "prompt_text"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("Manifest must include columns: prompt,prompt_text")
            for i, row in enumerate(reader):
                prompt_path = Path(row["prompt"]).expanduser().resolve()
                if not prompt_path.is_file():
                    raise FileNotFoundError(f"Prompt missing: {prompt_path}")
                name = row.get("name") or prompt_path.stem or f"prompt_{i+1}"
                cases.append(PromptCase(path=prompt_path, prompt_text=row["prompt_text"], name=name))
        return cases

    if not args.prompt:
        raise ValueError("Pass --prompt (repeatable) or --manifest.")

    prompt_paths = [Path(p).expanduser().resolve() for p in args.prompt]
    for p in prompt_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Prompt missing: {p}")

    prompt_texts: list[str] = []
    if args.prompt_text:
        if len(args.prompt_text) == 1:
            prompt_texts = [args.prompt_text[0] for _ in prompt_paths]
        elif len(args.prompt_text) == len(prompt_paths):
            prompt_texts = list(args.prompt_text)
        else:
            raise ValueError("Pass one --prompt-text for all prompts or one per prompt.")
    else:
        if asr_pipe is None:
            raise RuntimeError(
                "No --prompt-text provided and no local ASR cache found. "
                "Pass --prompt-text explicitly."
            )
        for p in prompt_paths:
            prompt_texts.append(asr_pipe(str(p))["text"].strip())

    for i, (p, t) in enumerate(zip(prompt_paths, prompt_texts)):
        cases.append(PromptCase(path=p, prompt_text=t, name=f"prompt_{i+1}_{p.stem}"))
    return cases


def _evaluate_quality(
    wav_np: np.ndarray,
    target_text: str,
    asr_pipe,
) -> dict[str, float | None] | tuple[dict[str, float | None], str]:
    wav_np = wav_np.astype(np.float32, copy=False).reshape(-1)
    peak = float(np.max(np.abs(wav_np))) if wav_np.size else 0.0
    clip_frac = float(np.mean(np.abs(wav_np) >= 0.999)) if wav_np.size else 0.0
    rms = _rms(wav_np)

    sr = 48000
    start = _slice_safe(wav_np, 0, int(0.12 * sr))
    start_body = _slice_safe(wav_np, int(0.12 * sr), int(0.52 * sr))
    tail = _slice_safe(wav_np, max(0, wav_np.size - int(0.12 * sr)), wav_np.size)
    tail_body = _slice_safe(
        wav_np,
        max(0, wav_np.size - int(0.52 * sr)),
        max(0, wav_np.size - int(0.12 * sr)),
    )
    start_ratio = _rms(start) / max(_rms(start_body), 1.0e-6)
    tail_ratio = _rms(tail) / max(_rms(tail_body), 1.0e-6)

    hyp_text = ""
    asr_sim = None
    repeat_ratio = None
    asr_words = None
    if asr_pipe is not None:
        hyp_text = asr_pipe({"sampling_rate": sr, "raw": wav_np})["text"]
        repeat_ratio = _repeat_ratio(hyp_text)
        asr_words = float(len(_normalize_text(hyp_text).split()))
        asr_sim = SequenceMatcher(
            None,
            _normalize_text(target_text),
            _normalize_text(hyp_text),
        ).ratio()

    metrics = {
        "peak": peak,
        "clip_frac": clip_frac,
        "rms": rms,
        "start_ratio": start_ratio,
        "tail_ratio": tail_ratio,
        "asr_sim": float(asr_sim) if asr_sim is not None else None,
        "repeat_ratio": float(repeat_ratio) if repeat_ratio is not None else None,
        "asr_words": float(asr_words) if asr_words is not None else None,
    }
    return metrics, hyp_text


def _wav_to_numpy(wav_obj) -> np.ndarray:
    if torch.is_tensor(wav_obj):
        return wav_obj.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    return np.asarray(wav_obj, dtype=np.float32).reshape(-1)


def _serialize_candidate(record: dict[str, Any], keep_asr_text: bool) -> dict[str, Any]:
    out = {
        "candidate_index": record.get("candidate_index"),
        "backend": record["backend"],
        "output_wav": record.get("output_wav"),
        "params": record["params"],
        "metrics": record["metrics"],
        "score": record["score"],
    }
    if keep_asr_text:
        out["asr_text"] = record.get("asr_text", "")
    return out


def _params_key(params: SynthesisParams) -> tuple[Any, ...]:
    return (
        params.num_steps,
        round(params.guidance_scale, 3),
        round(params.t_shift, 3),
        round(params.speed, 3),
        params.duration_pad_frames,
        round(params.ref_duration, 3),
        round(params.prompt_start, 3),
        round(params.prompt_fade_ms, 3),
        params.trim_silence,
        round(params.silence_threshold_db, 3),
        round(params.keep_silence_ms, 3),
        round(params.rms, 4),
        round(params.rms_min, 4),
        round(params.rms_max, 4),
        params.return_smooth,
    )


def _clamp_params(params: SynthesisParams, prompt_duration: float) -> SynthesisParams:
    p = replace(params)
    p.num_steps = int(min(10, max(3, p.num_steps)))
    p.guidance_scale = float(min(4.0, max(1.8, p.guidance_scale)))
    p.t_shift = float(min(0.68, max(0.32, p.t_shift)))
    p.speed = float(min(1.08, max(0.85, p.speed)))
    p.duration_pad_frames = int(min(56, max(0, p.duration_pad_frames)))
    p.prompt_fade_ms = float(min(32.0, max(4.0, p.prompt_fade_ms)))
    p.silence_threshold_db = float(min(-24.0, max(-56.0, p.silence_threshold_db)))
    p.keep_silence_ms = float(min(90.0, max(0.0, p.keep_silence_ms)))
    p.rms_max = float(min(0.04, max(0.016, p.rms_max)))
    p.rms_min = float(min(p.rms_max - 0.001, max(0.002, p.rms_min)))
    p.rms = float(min(p.rms_max, max(p.rms_min, p.rms)))

    max_start = max(0.0, prompt_duration - 1.2)
    p.prompt_start = float(min(max_start, max(0.0, p.prompt_start)))
    avail_ref = max(1.2, prompt_duration - p.prompt_start - 0.05)
    p.ref_duration = float(min(avail_ref, max(1.2, p.ref_duration)))
    return p


def _candidate_variants(
    base: SynthesisParams,
    round_idx: int,
    prompt_duration: float,
    thresholds: QualityThresholds,
    last_best_metrics: dict[str, float | None] | None,
    max_candidates: int,
) -> list[SynthesisParams]:
    candidates: list[SynthesisParams] = []
    seen: set[tuple[Any, ...]] = set()

    def add(candidate: SynthesisParams) -> None:
        clamped = _clamp_params(candidate, prompt_duration)
        key = _params_key(clamped)
        if key in seen:
            return
        seen.add(key)
        candidates.append(clamped)

    add(base)

    max_start = max(0.0, prompt_duration - 1.2)
    if round_idx == 1:
        add(replace(base, prompt_start=min(max_start, 0.18)))
        add(replace(base, prompt_start=min(max_start, 0.35), ref_duration=min(2.0, max(1.2, prompt_duration - 0.4))))
        add(replace(base, ref_duration=min(3.0, max(1.2, prompt_duration - 0.05))))
        add(replace(base, trim_silence=False, prompt_fade_ms=18.0))
        add(replace(base, silence_threshold_db=-48.0, keep_silence_ms=45.0))
        add(replace(base, guidance_scale=2.4, t_shift=0.45, speed=0.97, num_steps=max(6, base.num_steps)))
        add(replace(base, guidance_scale=2.0, t_shift=0.38, speed=1.0, return_smooth=False, num_steps=max(6, base.num_steps)))
    else:
        m = last_best_metrics or {}
        asr = float(m.get("asr_sim") or 0.0)
        repeat_ratio = float(m.get("repeat_ratio") or 1.0)
        start_ratio = float(m.get("start_ratio") or 0.0)
        tail_ratio = float(m.get("tail_ratio") or 0.0)
        clip_frac = float(m.get("clip_frac") or 0.0)
        peak = float(m.get("peak") or 0.0)

        if asr < thresholds.min_asr_sim or repeat_ratio > thresholds.max_repeat_ratio:
            add(replace(base, prompt_start=min(max_start, base.prompt_start + 0.16)))
            add(
                replace(
                    base,
                    prompt_start=min(max_start, base.prompt_start + 0.32),
                    ref_duration=max(1.2, base.ref_duration - 0.45),
                )
            )
            add(replace(base, trim_silence=not base.trim_silence))
            add(replace(base, return_smooth=not base.return_smooth))
            add(
                replace(
                    base,
                    guidance_scale=max(1.8, base.guidance_scale - 0.4),
                    t_shift=max(0.32, base.t_shift - 0.06),
                    speed=min(1.05, base.speed + 0.05),
                    num_steps=min(10, base.num_steps + 1),
                )
            )
            add(replace(base, silence_threshold_db=max(-56.0, base.silence_threshold_db - 3.0)))
            add(replace(base, guidance_scale=min(3.8, base.guidance_scale + 0.3), t_shift=min(0.62, base.t_shift + 0.06)))

        if start_ratio > thresholds.max_start_ratio:
            add(
                replace(
                    base,
                    prompt_start=min(max_start, base.prompt_start + 0.22),
                    prompt_fade_ms=min(30.0, base.prompt_fade_ms + 4.0),
                    silence_threshold_db=max(-56.0, base.silence_threshold_db - 2.0),
                )
            )

        if tail_ratio > thresholds.max_tail_ratio:
            add(
                replace(
                    base,
                    duration_pad_frames=min(56, base.duration_pad_frames + 10),
                    t_shift=max(0.32, base.t_shift - 0.05),
                    speed=min(1.05, base.speed + 0.03),
                )
            )

        if clip_frac > thresholds.max_clip_frac or peak > 0.98:
            add(replace(base, rms_max=max(0.018, base.rms_max - 0.003)))

        # Keep at least a couple mild exploration candidates each round.
        add(replace(base, speed=max(0.88, base.speed - 0.04), t_shift=min(0.62, base.t_shift + 0.05)))
        add(replace(base, guidance_scale=2.2, t_shift=0.42, speed=0.98))

    return candidates[: max(1, max_candidates)]


def _write_summary_markdown(
    out_path: Path,
    target_text: str,
    thresholds: QualityThresholds,
    prompt_reports: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Feedback Loop Report")
    lines.append("")
    lines.append(f"- Target text: `{target_text}`")
    lines.append(
        "- Thresholds: "
        f"ASR>={thresholds.min_asr_sim:.2f}, "
        f"start_ratio<={thresholds.max_start_ratio:.2f}, "
        f"tail_ratio<={thresholds.max_tail_ratio:.2f}, "
        f"clip_frac<={thresholds.max_clip_frac:.4f}, "
        f"repeat_ratio<={thresholds.max_repeat_ratio:.2f}"
    )
    lines.append("")

    for report in prompt_reports:
        final = report["final_choice"]
        lines.append(f"## {report['case_name']}")
        lines.append(f"- Prompt: `{report['prompt_path']}`")
        lines.append(f"- Final backend: `{final['backend']}`")
        lines.append(
            "- Final metrics: "
            f"asr={final['metrics'].get('asr_sim')}, "
            f"start={final['metrics']['start_ratio']:.3f}, "
            f"tail={final['metrics']['tail_ratio']:.3f}, "
            f"repeat={final['metrics'].get('repeat_ratio')}, "
            f"clip={final['metrics']['clip_frac']:.4f}, "
            f"peak={final['metrics']['peak']:.3f}"
        )
        lines.append(f"- Passed: `{report['passed']}` after {report['rounds_run']} rounds")
        lines.append(f"- Final params: `{json.dumps(final['params'], sort_keys=True)}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive MLX quality feedback loop with Torch-vs-MLX comparison.")
    parser.add_argument("--text", required=True, help="Target text to synthesize for every run.")
    parser.add_argument("--manifest", default=None, help="CSV with columns: prompt,prompt_text[,name].")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt path. Repeat for multiple prompts.")
    parser.add_argument(
        "--prompt-text",
        action="append",
        default=[],
        help="Prompt transcript. Provide once for all prompts or once per prompt.",
    )
    parser.add_argument("--model", default="YatharthS/LuxTTS", help="Model id or local path.")
    parser.add_argument("--device", default="mlx", help="Runtime device (recommended: mlx).")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads when device=cpu.")
    parser.add_argument(
        "--vocoder-set",
        choices=["both", "mlx", "torch"],
        default="both",
        help="When device=mlx, compare mlx+torch vocoders or force one.",
    )
    parser.add_argument("--max-rounds", type=int, default=4, help="Maximum tuning rounds per prompt.")
    parser.add_argument("--max-candidates-per-round", type=int, default=10, help="Parameter candidates explored each round.")
    parser.add_argument("--out-dir", default="feedback-loop-runs", help="Output directory.")
    parser.add_argument(
        "--keep-wavs",
        choices=["none", "final", "best-round", "all"],
        default="final",
        help="Storage policy for generated wavs (default: final).",
    )
    parser.add_argument(
        "--compact-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store compact JSON report (default: true).",
    )
    parser.add_argument(
        "--keep-asr-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw ASR transcript text in JSON report (default: false).",
    )
    parser.add_argument(
        "--clean-out-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete existing output directory before running (default: false).",
    )

    parser.add_argument("--min-asr-sim", type=float, default=0.78)
    parser.add_argument("--max-start-ratio", type=float, default=0.45)
    parser.add_argument("--max-tail-ratio", type=float, default=0.58)
    parser.add_argument("--max-clip-frac", type=float, default=0.0005)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.55)

    args = parser.parse_args()

    from luxtts_mlx import LuxTTS

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        min_asr_sim=args.min_asr_sim,
        max_start_ratio=args.max_start_ratio,
        max_tail_ratio=args.max_tail_ratio,
        max_clip_frac=args.max_clip_frac,
        max_repeat_ratio=args.max_repeat_ratio,
    )

    asr_pipe = _load_local_asr()
    if asr_pipe is None:
        print("Warning: local Whisper cache not found. ASR similarity scoring will be skipped.")

    cases = _load_cases(args, asr_pipe)
    print(f"Loaded {len(cases)} prompt cases.")

    engines: dict[str, Any] = {}
    if args.device == "mlx":
        vocoder_candidates = ["mlx", "torch"] if args.vocoder_set == "both" else [args.vocoder_set]
        for backend in vocoder_candidates:
            try:
                engines[backend] = LuxTTS(
                    args.model,
                    device="mlx",
                    threads=args.threads,
                    vocoder_backend=backend,
                )
                print(f"Loaded engine: device=mlx, vocoder={backend}")
            except Exception as ex:
                print(f"Warning: failed to load vocoder backend '{backend}': {ex}")
        if not engines:
            raise RuntimeError("No MLX engines loaded. Cannot run feedback loop.")
    else:
        engines[args.device] = LuxTTS(
            args.model,
            device=args.device,
            threads=args.threads,
        )
        print(f"Loaded engine: device={args.device}")

    prompt_reports: list[dict[str, Any]] = []
    target_text = args.text

    for case in cases:
        print(f"\n=== Case: {case.name} ({case.path}) ===")
        case_dir = out_dir / case.name
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_duration = _prompt_duration_seconds(case.path)

        params = SynthesisParams(ref_duration=min(2.4, max(1.2, prompt_duration - 0.05)))
        rounds: list[dict[str, Any]] = []
        passed = False
        last_best_metrics: dict[str, float | None] | None = None
        last_best_record: dict[str, Any] | None = None
        last_best_wav: np.ndarray | None = None

        for round_idx in range(1, args.max_rounds + 1):
            param_candidates = _candidate_variants(
                params,
                round_idx=round_idx,
                prompt_duration=prompt_duration,
                thresholds=thresholds,
                last_best_metrics=last_best_metrics,
                max_candidates=args.max_candidates_per_round,
            )
            per_round: list[dict[str, Any]] = []
            print(
                f"Round {round_idx}: testing {len(param_candidates)} param candidates "
                f"x {len(engines)} backend(s)"
            )

            best_record: dict[str, Any] | None = None
            best_wav: np.ndarray | None = None

            for cand_idx, candidate in enumerate(param_candidates, start=1):
                for backend_name, engine in engines.items():
                    round_tag = f"round{round_idx:02d}_cand{cand_idx:02d}_{backend_name}"
                    out_wav = case_dir / f"{round_tag}.wav"

                    encoded = engine.encode_prompt(
                        str(case.path),
                        duration=candidate.ref_duration,
                        rms=candidate.rms,
                        prompt_text=case.prompt_text,
                        offset=candidate.prompt_start,
                        fade_ms=candidate.prompt_fade_ms,
                        trim_silence=candidate.trim_silence,
                        silence_threshold_db=candidate.silence_threshold_db,
                        keep_silence_ms=candidate.keep_silence_ms,
                        rms_min=candidate.rms_min,
                        rms_max=candidate.rms_max,
                    )

                    wav = engine.generate_speech(
                        target_text,
                        encoded,
                        num_steps=candidate.num_steps,
                        guidance_scale=candidate.guidance_scale,
                        t_shift=candidate.t_shift,
                        speed=candidate.speed,
                        return_smooth=candidate.return_smooth,
                        duration_pad_frames=candidate.duration_pad_frames,
                    )
                    wav_np = _wav_to_numpy(wav)

                    metrics, hyp_text = _evaluate_quality(wav_np, target_text, asr_pipe)
                    score = _score_run(metrics)
                    output_wav = None
                    if args.keep_wavs == "all":
                        sf.write(out_wav, wav_np, 48000)
                        output_wav = str(out_wav)
                    record = {
                        "candidate_index": cand_idx,
                        "backend": backend_name,
                        "output_wav": output_wav,
                        "params": asdict(candidate),
                        "metrics": metrics,
                        "score": score,
                    }
                    if args.keep_asr_text:
                        record["asr_text"] = hyp_text
                    per_round.append(record)
                    print(
                        f"  cand={cand_idx:02d} {backend_name:>5} score={score:.3f} "
                        f"asr={metrics.get('asr_sim')} rep={metrics.get('repeat_ratio')} "
                        f"start={metrics['start_ratio']:.3f} tail={metrics['tail_ratio']:.3f} "
                        f"clip={metrics['clip_frac']:.4f}"
                    )

                    if best_record is None or score > best_record["score"]:
                        best_record = record
                        best_wav = wav_np

            if best_record is None:
                raise RuntimeError("No candidates evaluated in this round.")

            if args.keep_wavs == "best-round" and best_wav is not None:
                best_round_wav = case_dir / f"round{round_idx:02d}_best_{best_record['backend']}.wav"
                sf.write(best_round_wav, best_wav, 48000)
                best_record["output_wav"] = str(best_round_wav)

            best = best_record
            best_metrics = best["metrics"]
            if args.compact_report:
                round_report = {
                    "round": round_idx,
                    "seed_params": asdict(params),
                    "candidate_count": len(param_candidates),
                    "best": _serialize_candidate(best, keep_asr_text=args.keep_asr_text),
                }
            else:
                round_report = {
                    "round": round_idx,
                    "seed_params": asdict(params),
                    "candidate_count": len(param_candidates),
                    "candidates": [
                        _serialize_candidate(rec, keep_asr_text=args.keep_asr_text)
                        for rec in sorted(per_round, key=lambda x: x["score"], reverse=True)
                    ],
                    "best_backend": best["backend"],
                    "best_candidate_index": best.get("candidate_index"),
                    "best_score": best["score"],
                }
            rounds.append(round_report)
            last_best_record = dict(best)
            last_best_wav = best_wav

            if _passes_thresholds(best_metrics, thresholds):
                passed = True
                print(f"  -> thresholds passed on round {round_idx} with backend={best['backend']}")
                break

            params = _clamp_params(SynthesisParams(**best["params"]), prompt_duration)
            last_best_metrics = best_metrics
            print("  -> thresholds not met, best candidate becomes next seed params.")

        if last_best_record is None:
            raise RuntimeError("No final choice available.")

        if args.keep_wavs == "final" and last_best_wav is not None:
            final_wav_path = case_dir / f"final_best_{last_best_record['backend']}.wav"
            sf.write(final_wav_path, last_best_wav, 48000)
            last_best_record["output_wav"] = str(final_wav_path)
        elif args.keep_wavs == "none":
            last_best_record["output_wav"] = None

        final_choice = _serialize_candidate(last_best_record, keep_asr_text=args.keep_asr_text)
        report = {
            "case_name": case.name,
            "prompt_path": str(case.path),
            "prompt_text": case.prompt_text,
            "rounds_run": len(rounds),
            "passed": passed,
            "final_choice": final_choice,
            "rounds": rounds,
            "storage": {
                "keep_wavs": args.keep_wavs,
                "compact_report": args.compact_report,
                "keep_asr_text": args.keep_asr_text,
            },
        }
        prompt_reports.append(report)

        case_json = case_dir / "report.json"
        case_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    full_report = {
        "target_text": target_text,
        "thresholds": asdict(thresholds),
        "out_dir": str(out_dir),
        "storage": {
            "keep_wavs": args.keep_wavs,
            "compact_report": args.compact_report,
            "keep_asr_text": args.keep_asr_text,
        },
        "cases": prompt_reports,
    }
    (out_dir / "report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    _write_summary_markdown(out_dir / "SUMMARY.md", target_text, thresholds, prompt_reports)

    passed_count = sum(1 for r in prompt_reports if r["passed"])
    print(f"\nFinished: {passed_count}/{len(prompt_reports)} prompts passed thresholds.")
    print(f"Report: {out_dir / 'report.json'}")
    print(f"Summary: {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
