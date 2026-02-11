# Session Checkpoint (2026-02-11)

## Current State

- Package version: `0.3.2`
- Default runtime: full MLX (`--device mlx`, `--vocoder mlx`)
- Optional fallback still available: `--vocoder torch`

## What Works Well

- General English synthesis is clear.
- Cloning works reliably for short/clean prompt clips.
- Prompt-window controls (`--prompt-start`, `--ref-duration`) improve stability on long files.
- Prompt preprocessing now defaults to silence-edge trimming + safe RMS clamping.

## Known Remaining Issues

- Long/generated references can drift timbre/gender during long text.
- Some prompts may introduce light startup noise/haze.
- Loudness may feel high for some playback setups with `--output-peak 0.92`.

## Recommended Starting Preset

```bash
./.venv/bin/luxtts-mlx 'Your test sentence here.' \
  --prompt '/path/to/reference.wav' \
  --prompt-text 'Exact transcript of prompt clip.' \
  --prompt-start '0.8' \
  --ref-duration '2.4' \
  --prompt-fade-ms '18' \
  --trim-prompt-silence \
  --prompt-rms-min '0.006' \
  --prompt-rms-max '0.03' \
  --device 'mlx' \
  --vocoder 'mlx' \
  --num-steps '6' \
  --speed '1.0' \
  --t-shift '0.45' \
  --output-peak '0.8' \
  --out 'output.wav'
```

## Reset Workflow

1. Pull latest code and reinstall editable package.
2. Start with one known-good prompt and one sentence.
3. Tune only one parameter at a time (`prompt-start`, `ref-duration`, `speed`).
4. Expand to multi-voice smoke tests after single-voice quality is stable.
5. Use `tools/feedback_loop.py` for automated multi-round Torch-vs-MLX tuning.
6. Loop now explores multiple parameter candidates per round and penalizes repetition artifacts.
7. Use storage-safe defaults for long runs (`--keep-wavs final --compact-report`).
