import argparse
import sys

import numpy as np
import soundfile as sf

from luxtts_mlx import LuxTTS


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
        required=True,
        help="Path to prompt audio (wav/mp3).",
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
        "--return-smooth",
        action="store_true",
        help="Return smoother 24k output (uses secondary head).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    lux = LuxTTS(args.model, device=args.device, threads=args.threads)
    encoded = lux.encode_prompt(args.prompt, duration=args.ref_duration, rms=args.rms)
    wav = lux.generate_speech(
        args.text,
        encoded,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        speed=args.speed,
        return_smooth=args.return_smooth,
    )
    sf.write(args.out, np.array(wav).squeeze(), 48000)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
