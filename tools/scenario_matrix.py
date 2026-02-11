#!/usr/bin/env python3
"""
Run a fixed 5-scenario quality/performance matrix for LuxTTS.

This is intended for cross-machine smoke validation with storage-safe defaults.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
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
    prompt_text_source: str


@dataclass
class Scenario:
    scenario_id: str
    text: str


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _evaluate_quality(
    wav_np: np.ndarray,
    target_text: str,
    asr_pipe,
) -> tuple[dict[str, float | None], str]:
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
        try:
            asr_out = asr_pipe({"sampling_rate": sr, "raw": wav_np})
        except ValueError as ex:
            if "more than 3000 mel input features" in str(ex):
                asr_out = asr_pipe({"sampling_rate": sr, "raw": wav_np}, return_timestamps=True)
            else:
                print(f"Warning: ASR evaluation failed: {ex}")
                asr_out = {"text": ""}
        except Exception as ex:
            print(f"Warning: ASR evaluation failed: {ex}")
            asr_out = {"text": ""}

        hyp_text = str(asr_out.get("text", ""))
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


def _configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\].*",
        category=UserWarning,
    )


def _try_apply_linacodec_patch() -> None:
    try:
        from zipvoice.vocoder_patches import apply_linacodec_linkwitz_patch
    except Exception:
        return
    try:
        apply_linacodec_linkwitz_patch()
    except Exception:
        return


def _is_prompt_text_suspicious(
    prompt_text: str,
    max_words: int,
    max_repeat_ratio: float,
) -> tuple[bool, str]:
    norm = _normalize_text(prompt_text)
    if not norm:
        return True, "empty transcript"
    words = norm.split()
    if len(words) > max_words:
        return True, f"too many words ({len(words)} > {max_words})"
    repeat = _repeat_ratio(prompt_text)
    if repeat > max_repeat_ratio:
        return True, f"high repetition ratio ({repeat:.3f} > {max_repeat_ratio:.3f})"
    return False, ""


def _resolve_auto_prompt_text(
    *,
    path: Path,
    raw_text: str,
    target_text: str,
    policy: str,
    max_words: int,
    max_repeat_ratio: float,
) -> tuple[str, str]:
    cleaned = raw_text.strip()
    suspicious, reason = _is_prompt_text_suspicious(
        cleaned,
        max_words=max_words,
        max_repeat_ratio=max_repeat_ratio,
    )
    if not suspicious:
        return cleaned, "asr"
    if policy == "target-fallback":
        print(
            "Warning: auto prompt transcript looks unstable for "
            f"{path} ({reason}); falling back to target text."
        )
        return target_text, "target-fallback"
    if policy == "hello-fallback":
        print(
            "Warning: auto prompt transcript looks unstable for "
            f"{path} ({reason}); falling back to 'Hello.'."
        )
        return "Hello.", "hello-fallback"
    raise ValueError(
        "Auto prompt transcription looks unstable for "
        f"{path} ({reason}). Pass --prompt-text or set "
        "--auto-prompt-text-policy target-fallback."
    )


def _load_cases(args, asr_pipe) -> list[PromptCase]:
    if not args.prompt:
        raise ValueError("Pass at least one --prompt.")

    prompt_paths = [Path(p).expanduser().resolve() for p in args.prompt]
    valid_paths: list[Path] = []
    valid_texts: list[str] = []
    valid_sources: list[str] = []

    if args.prompt_text:
        if len(args.prompt_text) == 1:
            prompt_texts = [args.prompt_text[0] for _ in prompt_paths]
        elif len(args.prompt_text) == len(prompt_paths):
            prompt_texts = list(args.prompt_text)
        else:
            raise ValueError("Pass one --prompt-text for all prompts or one per prompt.")

        for p, t in zip(prompt_paths, prompt_texts):
            if not p.is_file():
                if args.skip_missing_prompts:
                    print(f"Warning: skipping missing prompt: {p}")
                    continue
                raise FileNotFoundError(f"Prompt missing: {p}")
            text = str(t).strip()
            if not text:
                raise ValueError(f"Prompt text is empty for prompt: {p}")
            valid_paths.append(p)
            valid_texts.append(text)
            valid_sources.append("arg")
    else:
        for p in prompt_paths:
            if p.is_file():
                valid_paths.append(p)
                continue
            if args.skip_missing_prompts:
                print(f"Warning: skipping missing prompt: {p}")
                continue
            raise FileNotFoundError(f"Prompt missing: {p}")

        if not valid_paths:
            raise ValueError("No valid prompt files found.")

        if asr_pipe is None:
            if args.auto_prompt_text_policy == "target-fallback":
                print("Warning: local Whisper cache not found; using target text as prompt text.")
                valid_texts = [args.fallback_prompt_text for _ in valid_paths]
                valid_sources = ["target-fallback" for _ in valid_paths]
            elif args.auto_prompt_text_policy == "hello-fallback":
                print("Warning: local Whisper cache not found; using 'Hello.' as prompt text.")
                valid_texts = ["Hello." for _ in valid_paths]
                valid_sources = ["hello-fallback" for _ in valid_paths]
            else:
                raise RuntimeError(
                    "No --prompt-text provided and no local ASR cache found. "
                    "Pass --prompt-text or set --auto-prompt-text-policy target-fallback."
                )
        else:
            for p in valid_paths:
                raw = asr_pipe(str(p))["text"]
                resolved, source = _resolve_auto_prompt_text(
                    path=p,
                    raw_text=raw,
                    target_text=args.fallback_prompt_text,
                    policy=args.auto_prompt_text_policy,
                    max_words=args.max_auto_prompt_words,
                    max_repeat_ratio=args.max_auto_prompt_repeat,
                )
                valid_texts.append(resolved)
                valid_sources.append(source)

    if not valid_paths:
        raise ValueError("No valid prompt files found.")

    cases: list[PromptCase] = []
    for i, (p, t, s) in enumerate(zip(valid_paths, valid_texts, valid_sources), start=1):
        cases.append(PromptCase(path=p, prompt_text=t, name=f"prompt_{i}_{p.stem}", prompt_text_source=s))
    return cases


def _default_scenarios() -> list[Scenario]:
    return [
        Scenario("short_plain", "Quick quality check for LuxTTS."),
        Scenario(
            "medium_conversational",
            "This is a medium length conversational sentence to validate cadence, articulation, and timbre stability.",
        ),
        Scenario(
            "long_paragraph",
            (
                "Today we are running a longer paragraph to evaluate whether the voice stays consistent from beginning to end, "
                "whether pauses feel natural, and whether final words remain intelligible without robotic trailing artifacts."
            ),
        ),
        Scenario(
            "numbers_symbols",
            "Order number 8472 ships on March twenty first, two thousand twenty six at five thirty p.m., total one hundred twenty nine dollars and ninety five cents.",
        ),
        Scenario(
            "punctuation_pauses",
            "Wait... are you sure? I mean, really sure; because this test mixes pauses, commas, and emphasis.",
        ),
    ]


def _write_summary(out_path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Scenario Matrix Report")
    lines.append("")
    lines.append(f"- Device: `{report['config']['device']}`")
    lines.append(f"- Vocoder set: `{report['config']['vocoder_set']}`")
    lines.append(f"- Scenarios: `{len(report['scenarios'])}`")
    lines.append(f"- Cases: `{len(report['cases'])}`")
    lines.append("")

    for case in report["cases"]:
        lines.append(f"## {case['case_name']}")
        lines.append(f"- Prompt: `{case['prompt_path']}`")
        lines.append(f"- Prompt text source: `{case['prompt_text_source']}`")
        lines.append("")
        for scenario in case["scenarios"]:
            best = scenario["best"]
            lines.append(f"### {scenario['scenario_id']}")
            lines.append(
                f"- Best backend: `{best['backend']}` score={best['score']:.3f} "
                f"asr={best['metrics'].get('asr_sim')} start={best['metrics']['start_ratio']:.3f} "
                f"tail={best['metrics']['tail_ratio']:.3f} repeat={best['metrics'].get('repeat_ratio')}"
            )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-scenario quality/performance matrix for LuxTTS.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt audio path. Repeat for multiple prompts.")
    parser.add_argument(
        "--prompt-text",
        action="append",
        default=[],
        help="Prompt transcript. Provide once for all prompts or once per prompt.",
    )
    parser.add_argument(
        "--fallback-prompt-text",
        default="This is a short prompt for LuxTTS testing.",
        help="Prompt text used when auto prompt transcription fallback is enabled.",
    )
    parser.add_argument("--model", default="YatharthS/LuxTTS", help="Model id or local path.")
    parser.add_argument("--device", default="mlx", help="Runtime device.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads when device=cpu.")
    parser.add_argument(
        "--vocoder-set",
        choices=["both", "mlx", "torch"],
        default="both",
        help="When device=mlx, compare mlx+torch vocoders or force one.",
    )
    parser.add_argument("--out-dir", default="scenario-matrix-runs", help="Output directory.")
    parser.add_argument(
        "--keep-wavs",
        choices=["none", "best", "all"],
        default="best",
        help="Wav retention policy (default: best per scenario).",
    )
    parser.add_argument(
        "--keep-asr-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include ASR hypothesis text in report (default: false).",
    )
    parser.add_argument(
        "--skip-missing-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing prompt files (default: true).",
    )
    parser.add_argument(
        "--auto-prompt-text-policy",
        choices=["strict", "target-fallback", "hello-fallback"],
        default="strict",
        help="Behavior when --prompt-text is omitted and auto transcript looks unstable.",
    )
    parser.add_argument("--max-auto-prompt-words", type=int, default=24)
    parser.add_argument("--max-auto-prompt-repeat", type=float, default=0.72)
    parser.add_argument(
        "--clean-out-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete existing output directory before running (default: false).",
    )

    # Generation defaults tuned for stable smoke checks.
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=2.6)
    parser.add_argument("--t-shift", type=float, default=0.45)
    parser.add_argument("--speed", type=float, default=0.96)
    parser.add_argument("--duration-pad-frames", type=int, default=20)
    parser.add_argument("--return-smooth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-duration", type=float, default=2.4)
    parser.add_argument("--prompt-start", type=float, default=0.0)
    parser.add_argument("--prompt-fade-ms", type=float, default=14.0)
    parser.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--silence-threshold-db", type=float, default=-44.0)
    parser.add_argument("--keep-silence-ms", type=float, default=35.0)
    parser.add_argument("--rms", type=float, default=0.01)
    parser.add_argument("--rms-min", type=float, default=0.006)
    parser.add_argument("--rms-max", type=float, default=0.03)

    args = parser.parse_args()
    _configure_runtime_warnings()
    _try_apply_linacodec_patch()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr_pipe = _load_local_asr()
    if asr_pipe is None:
        print("Warning: local Whisper cache not found. ASR similarity scoring will be skipped.")

    from luxtts_mlx import LuxTTS

    cases = _load_cases(args, asr_pipe)
    scenarios = _default_scenarios()
    print(f"Loaded {len(cases)} prompts and {len(scenarios)} scenarios.")

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
            raise RuntimeError("No MLX engines loaded. Cannot run matrix.")
    else:
        engines[args.device] = LuxTTS(
            args.model,
            device=args.device,
            threads=args.threads,
        )
        print(f"Loaded engine: device={args.device}")

    report_cases: list[dict[str, Any]] = []

    for case in cases:
        print(f"\n=== Case: {case.name} ({case.path}) ===")
        case_dir = out_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_report = {
            "case_name": case.name,
            "prompt_path": str(case.path),
            "prompt_text": case.prompt_text,
            "prompt_text_source": case.prompt_text_source,
            "scenarios": [],
        }

        encoded_cache: dict[str, Any] = {}
        for backend_name, engine in engines.items():
            t0 = time.perf_counter()
            encoded_cache[backend_name] = engine.encode_prompt(
                str(case.path),
                duration=args.ref_duration,
                rms=args.rms,
                prompt_text=case.prompt_text,
                offset=args.prompt_start,
                fade_ms=args.prompt_fade_ms,
                trim_silence=args.trim_silence,
                silence_threshold_db=args.silence_threshold_db,
                keep_silence_ms=args.keep_silence_ms,
                rms_min=args.rms_min,
                rms_max=args.rms_max,
            )
            print(f"  encoded prompt for {backend_name} in {time.perf_counter() - t0:.2f}s")

        for scenario in scenarios:
            print(f"  scenario={scenario.scenario_id}")
            backend_records: list[dict[str, Any]] = []
            best_rec: dict[str, Any] | None = None
            best_wav: np.ndarray | None = None

            for backend_name, engine in engines.items():
                gen_t0 = time.perf_counter()
                wav = engine.generate_speech(
                    scenario.text,
                    encoded_cache[backend_name],
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    t_shift=args.t_shift,
                    speed=args.speed,
                    return_smooth=args.return_smooth,
                    duration_pad_frames=args.duration_pad_frames,
                )
                gen_sec = time.perf_counter() - gen_t0
                wav_np = _wav_to_numpy(wav)
                metrics, hyp = _evaluate_quality(wav_np, scenario.text, asr_pipe)
                score = _score_run(metrics)

                rec: dict[str, Any] = {
                    "backend": backend_name,
                    "score": float(score),
                    "metrics": metrics,
                    "timing_sec": {"generate": float(gen_sec)},
                    "output_wav": None,
                }
                if args.keep_asr_text:
                    rec["asr_text"] = hyp

                if args.keep_wavs == "all":
                    wav_path = case_dir / f"{scenario.scenario_id}_{backend_name}.wav"
                    sf.write(wav_path, wav_np, 48000)
                    rec["output_wav"] = str(wav_path)

                backend_records.append(rec)
                if best_rec is None or rec["score"] > best_rec["score"]:
                    best_rec = rec
                    best_wav = wav_np

                print(
                    f"    {backend_name:>5} score={score:.3f} asr={metrics.get('asr_sim')} "
                    f"start={metrics['start_ratio']:.3f} tail={metrics['tail_ratio']:.3f} "
                    f"repeat={metrics.get('repeat_ratio')} time={gen_sec:.2f}s"
                )

            if best_rec is None:
                raise RuntimeError("No scenario candidates were evaluated.")

            if args.keep_wavs == "best" and best_wav is not None:
                best_path = case_dir / f"{scenario.scenario_id}_best_{best_rec['backend']}.wav"
                sf.write(best_path, best_wav, 48000)
                best_rec["output_wav"] = str(best_path)

            if args.keep_wavs == "none":
                best_rec["output_wav"] = None
                for rec in backend_records:
                    rec["output_wav"] = None

            case_report["scenarios"].append(
                {
                    "scenario_id": scenario.scenario_id,
                    "text": scenario.text,
                    "best": best_rec,
                    "results": backend_records,
                }
            )

        report_cases.append(case_report)

    report = {
        "config": {
            "device": args.device,
            "vocoder_set": args.vocoder_set,
            "keep_wavs": args.keep_wavs,
            "keep_asr_text": args.keep_asr_text,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "t_shift": args.t_shift,
            "speed": args.speed,
            "duration_pad_frames": args.duration_pad_frames,
            "return_smooth": args.return_smooth,
        },
        "scenarios": [asdict(s) for s in scenarios],
        "cases": report_cases,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(out_dir / "SUMMARY.md", report)

    print(f"\nReport: {out_dir / 'report.json'}")
    print(f"Summary: {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
