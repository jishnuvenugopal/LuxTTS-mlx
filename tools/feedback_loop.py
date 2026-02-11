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


def _score_run(metrics: dict[str, float | None]) -> float:
    asr = metrics.get("asr_sim")
    asr_term = 2.8 * float(asr if asr is not None else 0.75)
    return float(
        asr_term
        - 0.8 * float(metrics["start_ratio"])
        - 0.7 * float(metrics["tail_ratio"])
        - 8.0 * float(metrics["clip_frac"])
    )


def _passes_thresholds(metrics: dict[str, float | None], thresholds: QualityThresholds) -> bool:
    if metrics["clip_frac"] > thresholds.max_clip_frac:
        return False
    if metrics["start_ratio"] > thresholds.max_start_ratio:
        return False
    if metrics["tail_ratio"] > thresholds.max_tail_ratio:
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
    if asr_pipe is not None:
        hyp_text = asr_pipe({"sampling_rate": sr, "raw": wav_np})["text"]
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
    }
    return metrics, hyp_text


def _wav_to_numpy(wav_obj) -> np.ndarray:
    if torch.is_tensor(wav_obj):
        return wav_obj.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    return np.asarray(wav_obj, dtype=np.float32).reshape(-1)


def _next_params(
    current: SynthesisParams,
    metrics: dict[str, float | None],
    thresholds: QualityThresholds,
    prompt_duration: float,
) -> SynthesisParams:
    nxt = replace(current)
    max_start = max(0.0, prompt_duration - 1.2)

    if metrics["start_ratio"] > thresholds.max_start_ratio:
        nxt.prompt_start = min(max_start, nxt.prompt_start + 0.14)
        nxt.prompt_fade_ms = min(30.0, nxt.prompt_fade_ms + 3.0)
        nxt.silence_threshold_db = max(-54.0, nxt.silence_threshold_db - 2.0)

    if metrics["tail_ratio"] > thresholds.max_tail_ratio:
        nxt.duration_pad_frames = min(48, nxt.duration_pad_frames + 8)
        nxt.t_shift = max(0.35, nxt.t_shift - 0.05)
        nxt.speed = min(1.05, nxt.speed + 0.03)

    asr_sim = metrics.get("asr_sim")
    if asr_sim is not None and asr_sim < thresholds.min_asr_sim:
        nxt.num_steps = min(8, nxt.num_steps + 1)
        nxt.t_shift = max(0.35, nxt.t_shift - 0.03)
        nxt.speed = min(1.05, max(0.95, nxt.speed) + 0.01)
        nxt.ref_duration = min(nxt.ref_duration + 0.35, max(1.4, prompt_duration - nxt.prompt_start - 0.05))

    if metrics["clip_frac"] > thresholds.max_clip_frac or metrics["peak"] > 0.98:
        nxt.rms_max = max(0.018, nxt.rms_max - 0.003)
        nxt.rms = min(nxt.rms, nxt.rms_max)

    available_ref = max(1.2, prompt_duration - nxt.prompt_start - 0.05)
    nxt.ref_duration = min(max(1.2, nxt.ref_duration), available_ref)
    if nxt.rms_min >= nxt.rms_max:
        nxt.rms_min = max(0.002, nxt.rms_max - 0.004)
    return nxt


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
        f"clip_frac<={thresholds.max_clip_frac:.4f}"
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
    parser.add_argument("--out-dir", default="feedback-loop-runs", help="Output directory.")

    parser.add_argument("--min-asr-sim", type=float, default=0.78)
    parser.add_argument("--max-start-ratio", type=float, default=0.45)
    parser.add_argument("--max-tail-ratio", type=float, default=0.58)
    parser.add_argument("--max-clip-frac", type=float, default=0.0005)

    args = parser.parse_args()

    from luxtts_mlx import LuxTTS

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        min_asr_sim=args.min_asr_sim,
        max_start_ratio=args.max_start_ratio,
        max_tail_ratio=args.max_tail_ratio,
        max_clip_frac=args.max_clip_frac,
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
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_duration = _prompt_duration_seconds(case.path)

        params = SynthesisParams(ref_duration=min(2.4, max(1.2, prompt_duration - 0.05)))
        rounds: list[dict[str, Any]] = []
        passed = False

        for round_idx in range(1, args.max_rounds + 1):
            per_backend: list[dict[str, Any]] = []
            print(f"Round {round_idx}: testing {len(engines)} backend(s)")

            for backend_name, engine in engines.items():
                round_tag = f"round{round_idx:02d}_{backend_name}"
                out_wav = case_dir / f"{round_tag}.wav"

                encoded = engine.encode_prompt(
                    str(case.path),
                    duration=params.ref_duration,
                    rms=params.rms,
                    prompt_text=case.prompt_text,
                    offset=params.prompt_start,
                    fade_ms=params.prompt_fade_ms,
                    trim_silence=params.trim_silence,
                    silence_threshold_db=params.silence_threshold_db,
                    keep_silence_ms=params.keep_silence_ms,
                    rms_min=params.rms_min,
                    rms_max=params.rms_max,
                )

                wav = engine.generate_speech(
                    target_text,
                    encoded,
                    num_steps=params.num_steps,
                    guidance_scale=params.guidance_scale,
                    t_shift=params.t_shift,
                    speed=params.speed,
                    return_smooth=params.return_smooth,
                    duration_pad_frames=params.duration_pad_frames,
                )
                wav_np = _wav_to_numpy(wav)
                sf.write(out_wav, wav_np, 48000)

                metrics, hyp_text = _evaluate_quality(wav_np, target_text, asr_pipe)
                score = _score_run(metrics)
                record = {
                    "backend": backend_name,
                    "output_wav": str(out_wav),
                    "params": asdict(params),
                    "metrics": metrics,
                    "score": score,
                    "asr_text": hyp_text,
                }
                per_backend.append(record)
                print(
                    f"  {backend_name:>5} score={score:.3f} "
                    f"asr={metrics.get('asr_sim')} "
                    f"start={metrics['start_ratio']:.3f} "
                    f"tail={metrics['tail_ratio']:.3f} clip={metrics['clip_frac']:.4f}"
                )

            best = max(per_backend, key=lambda x: x["score"])
            best_metrics = best["metrics"]
            round_report = {
                "round": round_idx,
                "params": asdict(params),
                "candidates": per_backend,
                "best_backend": best["backend"],
                "best_score": best["score"],
            }
            rounds.append(round_report)

            if _passes_thresholds(best_metrics, thresholds):
                passed = True
                print(f"  -> thresholds passed on round {round_idx} with backend={best['backend']}")
                break

            params = _next_params(params, best_metrics, thresholds, prompt_duration)
            print("  -> thresholds not met, tuning params and continuing.")

        final_choice = max(rounds[-1]["candidates"], key=lambda x: x["score"])
        report = {
            "case_name": case.name,
            "prompt_path": str(case.path),
            "prompt_text": case.prompt_text,
            "rounds_run": len(rounds),
            "passed": passed,
            "final_choice": final_choice,
            "rounds": rounds,
        }
        prompt_reports.append(report)

        case_json = case_dir / "report.json"
        case_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    full_report = {
        "target_text": target_text,
        "thresholds": asdict(thresholds),
        "out_dir": str(out_dir),
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
