import argparse
import logging
import os
import sys
import warnings

import numpy as np
import soundfile as sf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate speech with LuxTTS (MLX port).",
    )
    parser.add_argument(
        "--text",
        required=True,
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
        action="store_true",
        help="Return smoother 24k output (uses secondary head).",
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
        prompt_text = args.text

    lux = LuxTTS(args.model, device=args.device, threads=args.threads)
    if prompt_path is None:
        print("No prompt audio provided; using a synthetic prompt for a quick smoke test.")
    encoded = lux.encode_prompt(
        prompt_path,
        duration=args.ref_duration,
        rms=args.rms,
        prompt_text=prompt_text,
    )
    wav = lux.generate_speech(
        args.text,
        encoded,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        speed=args.speed,
        return_smooth=args.return_smooth,
    )
    sf.write(out_path, np.array(wav).squeeze(), 48000)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
