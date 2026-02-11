# Session Checkpoint (2026-02-11)

## Mission

Stabilize LuxTTS quality and consistency with the least practical steps, while keeping:

- MLX stack and Torch stack independently viable.
- Recursive feedback runs storage-safe.
- A repeatable scenario matrix for regression checks.

---

## Repos And Runtime Context

- Primary working repo (authoring): `/Users/jv/Downloads/Codex/lux-tts/LuxTTS-mlx`
- User execution repo (active testing): `/Users/jv/Downloads/LuxTTS-mlx-test`
- Python env used for runs: `/Users/jv/Downloads/LuxTTS-mlx-test/.venv/bin/python`
- Date context: `2026-02-11`

---

## Critical Input/Prompt Paths Used

Source prompt files:

- `/Users/jv/Downloads/prompt.wav`
- `/Users/jv/Downloads/kyutai/pytorch_same_text.wav`
- `/Users/jv/Downloads/kyutai/mlx_same_text.wav`
- `/Users/jv/Downloads/test-mlx-package/test_output.wav`

Convenience symlink folder created for loop commands:

- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice1_prompt.wav`
- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice2_kyutai_pytorch.wav`
- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice3_kyutai_mlx.wav`
- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice4_testpkg.wav`

---

## Code Changes Completed

### 1) Feedback Loop Hardening

File: `/Users/jv/Downloads/LuxTTS-mlx-test/tools/feedback_loop.py`

Implemented:

- Storage controls:
  - `--round-history {none,best,all}`
  - `--write-case-reports/--no-write-case-reports`
- Missing file resilience:
  - `--skip-missing-prompts` (default true)
- Prompt transcript safety:
  - `--auto-prompt-text-policy {strict,target-fallback,hello-fallback}`
  - `--max-auto-prompt-words`
  - `--max-auto-prompt-repeat`
  - Reject or fallback when auto prompt transcript is obviously unstable.
- Performance optimization:
  - Prompt-encoding cache across candidate variants.
  - Added per-candidate timing and cache-hit metadata in report.
- ASR long-form robustness:
  - If Whisper throws `>3000 mel features` error, retry with `return_timestamps=True`.
- Warning cleanup:
  - Suppress known noisy LinaCodec resize warning.
  - Suppress deprecated autocast warning from vocoder dependency.

### 2) 5-Scenario Matrix Runner Added

New file: `/Users/jv/Downloads/LuxTTS-mlx-test/tools/scenario_matrix.py`

Capabilities:

- Fixed 5 scenarios:
  - `short_plain`
  - `medium_conversational`
  - `long_paragraph`
  - `numbers_symbols`
  - `punctuation_pauses`
- Supports multiple prompts and MLX/Torch comparison (`--vocoder-set both` when `--device mlx`).
- Storage-safe wave retention (`--keep-wavs none|best|all`).
- Includes same prompt-text safety and long-form ASR fallback behavior.
- Produces:
  - `report.json`
  - `SUMMARY.md`
  - scenario-level best WAVs (if enabled)

### 3) LinaCodec Linkwitz Patch Scope Improved

File: `/Users/jv/Downloads/LuxTTS-mlx-test/zipvoice/vocoder_patches.py`

- Patch now also rewrites any already-imported `linacodec.vocoder.*` module aliases that copied the old function symbol.
- Reduces warning/leak-through chance from import-order issues.

### 4) Documentation Updated

- `/Users/jv/Downloads/LuxTTS-mlx-test/README.md`
- `/Users/jv/Downloads/LuxTTS-mlx-test/docs/SESSION_CHECKPOINT.md` (this file)

---

## Test Runs And Artifacts

### A) Feedback Loop Baseline (v3)

Artifacts:

- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v3/report.json`
- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v3/SUMMARY.md`

Findings:

- All 4 prompts failed strict thresholds.
- Best practical prompts were prompt 1 and 2.
- Backend often converged to torch in final picks.

### B) Feedback Loop Storage-Optimized (v4)

Artifacts:

- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v4/report.json`
- `/Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v4/SUMMARY.md`

Storage outcome:

- `report.json` reduced to ~9 KB with `--round-history none` and `--no-write-case-reports`.

Key result:

- Best single candidate in this run was:
  - `/Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v4/prompt_2_pytorch_same_text/final_best_mlx.wav`

### C) Scenario Matrix (v1, MLX+Torch comparison)

Artifacts:

- `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/report.json`
- `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/SUMMARY.md`

Result summary:

- 2 prompts x 5 scenarios = 10 best-of-scenario outcomes.
- Backend wins split: MLX 5, Torch 5.
- Practical quality gate pass rate was low (~2/10 under ASR+repeat+start+tail gate).
- Long/numeric/punctuation scenarios still unstable.

Top two listen-first files from matrix:

- `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/prompt_1_prompt/medium_conversational_best_mlx.wav`
- `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/prompt_2_pytorch_same_text/short_plain_best_torch.wav`

---

## Errors Encountered And Fix Status

### 1) Missing prompt path failures

Error:

- `FileNotFoundError: Prompt missing ...`

Status:

- Fixed by `--skip-missing-prompts` and creating `voice-smoke` symlinks.

### 2) LinaCodec warning spam

Warning:

- `An output with one or more elements was resized since it had shape [] ...`

Status:

- Addressed by patch application + warning suppression.

### 3) Scenario matrix crash on long-form ASR

Error:

- `ValueError: ... more than 3000 mel input features ... requires return_timestamps=True`

Status:

- Fixed in both scripts by retry with `return_timestamps=True`.

### 4) Deprecation warning from vocoder autocast

Warning:

- ``torch.cuda.amp.autocast(args...)` is deprecated`

Status:

- Suppressed in tool runtime warning filters.

---

## Root-Cause Assessment (Current)

Most remaining quality problems are shared-stack issues, not a single backend bug:

1. **Duration/length prediction heuristic** is brittle for long/numeric/punctuation-heavy text.
2. **Prompt/text conditioning sensitivity** still high for noisy or mismatched prompt text.
3. **Vocoder backend differences** are secondary for repetition/content drift; they mostly affect texture/timbre.

Conclusion:

- Not realistic to claim “any prompt on any machine” today.
- Realistic to achieve strong reliability with one more targeted pass.

---

## Minimum-Step Plan For Next Thread

Goal: make both independent stacks pass the same gate.

1. Enforce strict production prompt text policy:
   - require explicit `--prompt-text` in prod path (or strict fallback policy).
2. Implement long-text chunked generation in shared path:
   - sentence/phrase chunking + short crossfade.
3. Test true independent stacks (not mixed fallback):
   - Full MLX stack: `--device mlx --vocoder-set mlx`
   - Full Torch stack: run on torch device path (`--device mps` or `--device cuda` where available).
4. Gate release on scenario matrix pass criteria for both stacks.

---

## Ready-To-Run Commands (Known Good Entrypoints)

### Feedback loop (storage-safe)

```bash
cd '/Users/jv/Downloads/LuxTTS-mlx-test'

./.venv/bin/python tools/feedback_loop.py \
  --text "This is a stability smoke test for LuxTTS MLX prompt preprocessing." \
  --prompt /Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice1_prompt.wav \
  --prompt /Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice2_kyutai_pytorch.wav \
  --prompt /Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice3_kyutai_mlx.wav \
  --prompt /Users/jv/Downloads/LuxTTS-mlx-test/voice-smoke/voice4_testpkg.wav \
  --device mlx \
  --vocoder-set both \
  --max-rounds 6 \
  --max-candidates-per-round 12 \
  --keep-wavs final \
  --round-history none \
  --compact-report \
  --no-keep-asr-text \
  --no-write-case-reports \
  --clean-out-dir \
  --out-dir /Users/jv/Downloads/LuxTTS-mlx-test/voice-feedback-loop-v4
```

### Scenario matrix (current regression gate)

```bash
cd '/Users/jv/Downloads/LuxTTS-mlx-test'

./.venv/bin/python tools/scenario_matrix.py \
  --prompt /Users/jv/Downloads/prompt.wav \
  --prompt-text "This is a short prompt for LuxTTS testing." \
  --prompt /Users/jv/Downloads/kyutai/pytorch_same_text.wav \
  --prompt-text "Hello, this is a test." \
  --device mlx \
  --vocoder-set both \
  --keep-wavs best \
  --clean-out-dir \
  --out-dir /Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1
```

---

## Thread Restart Instructions

If starting a fresh thread, begin with:

1. Read this file first.
2. Read:
   - `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/SUMMARY.md`
   - `/Users/jv/Downloads/LuxTTS-mlx-test/scenario-matrix-v1/report.json`
3. Continue with the minimum-step plan above (strict prompt text + chunked generation + dual-stack gate).

