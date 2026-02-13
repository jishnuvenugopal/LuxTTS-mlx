# LuxTTS-MLX Port Plan

Updated: 2026-02-13

## 1) Upstream Baseline

- Upstream repo: `ysharma3501/LuxTTS`
- Upstream head: `34820963ee97f406619e5983771e572f779a600a` (2026-01-28)
- Sync state of this fork (`master`): `0` behind, `42` ahead

Conclusion: there are no new upstream commits to merge right now. Current work should focus on MLX quality and parity polish.

## 2) Current Quality Findings

From recent voice smoke tests:

- Voice is now intelligible in English on MLX.
- Some prompts still show startup noise or slight haze.
- Some outputs end with robotic tails.
- Cross-prompt consistency is still sensitive to reference quality.
- Short/generated references can drift gender/timbre on long generations.

## 3) Plan To Reach "Pocket-TTS-MLX" Smoothness

### Phase A: Parity Hardening

- Keep `tools/compare_mlx_torch.py` as the parity gate for:
  - `text_embed`
  - `text_condition`
  - `fm_decoder` outputs
- Add strict target thresholds and fail CI when drift grows.
- Validate MLX/Torch parity with both CPU and Apple GPU runs.

### Phase B: Inference Quality Stabilization

- Keep smooth output path default (`return_smooth=True`).
- Tune defaults for clarity and pacing:
  - `speed` default around `0.90-0.95`
  - small start/end padding to improve first/last word clarity
- Add optional prompt pre-processing:
  - trim leading/trailing silence
  - clamp prompt RMS into a safe range
  - Status: implemented in CLI defaults (`--trim-prompt-silence`, `--prompt-rms-min`, `--prompt-rms-max`)

### Phase C: CLI/Product UX

- Preserve one-liner UX:
  - `luxtts-mlx 'Hello from MLX!' --out 'output.wav' --device 'mlx'`
- Keep prompt usage explicit and beginner-safe:
  - clear error messages for missing prompt files
  - clear warning when optional phonemizer extras are missing
- Keep `phonemize` extra documented as the recommended English setup.

### Phase D: Release Gating

- Build a small voice smoke matrix (4+ prompt clips, same text).
- Gate release on intelligibility and artifact checks.
- Publish only after all smoke clips pass acceptable quality.
- Automate iterative tuning + backend comparison with `tools/feedback_loop.py`.

## 4) Versioning Strategy

- `0.1.0`: initial stable public release on GitHub and PyPI (`LuxTTS-mlx`).
- `0.1.1` (next): post-release quality pass focused on strict prompt-text path and long-text chunking/crossfade.
- `0.2.0` (target milestone): stronger dual-stack quality gate coverage across MLX-only and Torch-only paths.

Historical note:

- Earlier internal iteration labels referenced `0.2.x` and `0.3.x` feature bundles before the public version reset to `0.1.0`.
