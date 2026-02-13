# Changelog

## v0.1.0 (2026-02-13)

Initial stable package release of `LuxTTS-mlx`.

- GitHub release: `https://github.com/jishnuvenugopal/LuxTTS-mlx/releases/tag/v0.1.0`
- PyPI release: `https://pypi.org/project/LuxTTS-mlx/0.1.0/`

- Ships `luxtts-mlx` CLI and Python API (`luxtts_mlx.LuxTTS`).
- Supports MLX runtime on Apple Silicon with optional torch vocoder fallback.
- Includes prompt preprocessing controls for improved stability:
  - prompt start windowing
  - prompt fade at clip edges
  - optional prompt silence trimming
  - safe prompt RMS clamping
- Includes quality/regression tooling:
  - `tools/feedback_loop.py`
  - `tools/scenario_matrix.py`
- Documents current known quality sensitivities and recommended smoke checks.
- PyPI-compatible packaging metadata (no direct VCS dependency in `Requires-Dist`);
  install LinaCodec separately via git.

Known limitations for this initial stable:

- Long-form, numbers-heavy, and punctuation-heavy text can still be less stable.
- Best quality remains sensitive to prompt cleanliness and prompt-text accuracy.
