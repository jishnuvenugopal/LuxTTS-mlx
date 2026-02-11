import argparse
import logging
import os
import re
import sys
import warnings

import numpy as np
import soundfile as sf


def _contains_english(text: str | None) -> bool:
    if not text:
        return False
    return re.search(r"[A-Za-z]", text) is not None


def _ensure_phonemizer_for_english(text: str, prompt_text: str | None) -> tuple[bool, str | None]:
    if not (_contains_english(text) or _contains_english(prompt_text)):
        return True, None
    try:
        from zipvoice.tokenizer.tokenizer import _get_phonemizer, _phonemizer_install_hint
    except Exception as ex:
        return False, f"Unable to check phonemizer dependency: {ex}"

    phonemizer, err = _get_phonemizer()
    if phonemizer is None or err is not None:
        return False, _phonemizer_install_hint()
    return True, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate speech with LuxTTS (MLX port).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize (positional).",
    )
    parser.add_argument(
        "--text",
        dest="text_arg",
        default=None,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to prompt audio (wav/mp3). Optional for quick tests.",
    )
    parser.add_argument(
        "--out",
        default="output.wav",
        help="Output wav path (default: output.wav).",
    )
    parser.add_argument(
        "--model",
        default="YatharthS/LuxTTS",
        help="Model id or local path (default: YatharthS/LuxTTS).",
    )
    parser.add_argument(
        "--device",
        default="mlx",
        help="Device: mlx/cpu/cuda/mps (default: mlx).",
    )
    parser.add_argument(
        "--vocoder",
        default="mlx",
        choices=["mlx", "torch"],
        help="Vocoder backend when device=mlx (default: mlx).",
    )
    parser.add_argument(
        "--vocoder-device",
        default=None,
        help="Torch vocoder device when --vocoder torch (default: mps if available, else cpu).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads when using device=cpu (default: 4).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help="Sampling steps (default: 4).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (default: 3.0).",
    )
    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Sampling temperature-like parameter (default: 0.5).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed factor (default: 1.0).",
    )
    parser.add_argument(
        "--lead-silence-ms",
        type=int,
        default=120,
        help="Silence to prepend in milliseconds (default: 120).",
    )
    parser.add_argument(
        "--tail-silence-ms",
        type=int,
        default=180,
        help="Silence to append in milliseconds (default: 180).",
    )
    parser.add_argument(
        "--rms",
        type=float,
        default=0.01,
        help="Prompt RMS loudness (default: 0.01).",
    )
    parser.add_argument(
        "--ref-duration",
        type=float,
        default=5.0,
        help="Prompt duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Override prompt transcription (skips Whisper).",
    )
    parser.add_argument(
        "--return-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use smoother 24k output path (default: true). "
            "Pass --no-return-smooth for the sharper 48k path."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and warnings.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    text = args.text_arg or args.text
    if not text:
        print("Error: text is required (pass it positionally or with --text).", file=sys.stderr)
        return 2
    if not args.verbose:
        os.environ.setdefault("LUXTTS_SUPPRESS_OPTIONAL_WARNINGS", "1")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore",
            message="PySoundFile failed.*",
            module="librosa",
        )
        logging.getLogger().setLevel(logging.ERROR)
        try:
            from transformers.utils import logging as hf_logging
        except Exception:
            hf_logging = None
        if hf_logging is not None:
            hf_logging.set_verbosity_error()
        try:
            from huggingface_hub.utils import logging as hf_hub_logging
        except Exception:
            hf_hub_logging = None
        if hf_hub_logging is not None:
            hf_hub_logging.set_verbosity_error()

    from luxtts_mlx import LuxTTS

    prompt_path = os.path.expanduser(args.prompt) if args.prompt else None
    out_path = os.path.expanduser(args.out)

    if prompt_path is not None and not os.path.isfile(prompt_path):
        print(f"Error: prompt audio not found: {prompt_path}", file=sys.stderr)
        print("Pass a valid path, e.g. --prompt /full/path/to/prompt.wav", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    prompt_text = args.prompt_text
    if prompt_path is None and not prompt_text:
        prompt_text = text

    ok, err = _ensure_phonemizer_for_english(text, prompt_text)
    if not ok:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    lux = LuxTTS(
        args.model,
        device=args.device,
        threads=args.threads,
        vocoder_backend=args.vocoder,
        vocoder_device=args.vocoder_device,
    )
    if prompt_path is None:
        print("No prompt audio provided; using a synthetic prompt for a quick smoke test.")
    encoded = lux.encode_prompt(
        prompt_path,
        duration=args.ref_duration,
        rms=args.rms,
        prompt_text=prompt_text,
    )
    try:
        wav = lux.generate_speech(
            text,
            encoded,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            t_shift=args.t_shift,
            speed=args.speed,
            return_smooth=args.return_smooth,
        )
    except RuntimeError as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 2
    wav_np = np.array(wav).squeeze()
    if wav_np.ndim != 1:
        wav_np = wav_np.reshape(-1)

    sample_rate = 48000
    lead_samples = max(0, int(args.lead_silence_ms * sample_rate / 1000))
    tail_samples = max(0, int(args.tail_silence_ms * sample_rate / 1000))
    if lead_samples > 0:
        wav_np = np.concatenate([np.zeros(lead_samples, dtype=wav_np.dtype), wav_np])
    if tail_samples > 0:
        wav_np = np.concatenate([wav_np, np.zeros(tail_samples, dtype=wav_np.dtype)])

    sf.write(out_path, wav_np, sample_rate)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
