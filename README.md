# LuxTTS
<p align="center">
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Colab Notebook">
  </a>
</p>

LuxTTS is an lightweight zipvoice based text-to-speech model designed for high quality voice cloning and realistic generation at speeds exceeding 150x realtime.

https://github.com/user-attachments/assets/a3b57152-8d97-43ce-bd99-26dc9a145c29

### Upstream sync status (2026-02-11)
- Upstream LuxTTS commit synced: `34820963ee97f406619e5983771e572f779a600a` (2026-01-28).
- Local MLX fork is `0` commits behind upstream on `master`.
- Ongoing MLX quality and parity plan: `docs/MLX_PORT_PLAN.md`.


### The main features are
- Voice cloning: SOTA voice cloning on par with models 10x larger.
- Clarity: Clear 48khz speech generation unlike most TTS models which are limited to 24khz.
- Speed: Reaches speeds of 150x realtime on a single GPU and faster then realtime on CPU's as well.
- Efficiency: Fits within 1gb vram meaning it can fit in any local gpu.

## Usage
You can try it locally, colab, or spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YatharthS/LuxTTS)

#### Simple installation:
```
pip install LuxTTS-mlx
```

#### Recommended for English synthesis (includes phonemizer):
```
pip install "LuxTTS-mlx[phonemize]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html
```

#### From source:
```
git clone https://github.com/jishnuvenugopal/LuxTTS-mlx.git
cd LuxTTS-mlx
pip install -r requirements.txt
```

#### Load model:
```python
from luxtts_mlx import LuxTTS

# load model on GPU
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# load model on CPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

# load model on MPS for macs
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')

# load model on MLX (Apple Silicon, Python 3.11+)
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='mlx')
```

> Note: On MLX, both diffusion and vocoder run in MLX by default.

> Note: The MLX vocoder path uses `vocos-mlx` and will download the LuxTTS vocoder weights on first run.

> Important: English synthesis requires `piper_phonemize`.
> Install with: `pip install "LuxTTS-mlx[phonemize]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html`

#### Simple inference
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```

#### Inference with sampling params:
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

rms = 0.01 ## higher makes it sound louder(0.01 or so recommended)
t_shift = 0.9 ## sampling param, higher can sound better but worse WER
num_steps = 4 ## sampling param, higher sounds better but takes longer(3-4 is best for efficiency)
speed = 1.0 ## sampling param, controls speed of audio(lower=slower)
return_smooth = True ## sampling param, smoother/clearer default output path
ref_duration = 5 ## Setting it lower can speedup inference, set to 1000 if you find artifacts.

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=ref_duration, rms=rms)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=num_steps, t_shift=t_shift, speed=speed, return_smooth=return_smooth)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```
## Tips
- Please use at minimum a 3 second audio file for voice cloning.
- Best cloning quality comes from clean real speech (single speaker, no music/noise), typically 3-8 seconds.
- `return_smooth=True` is the default and usually sounds clearer; use `False` for sharper 48k output.
- Lower t_shift for less possible pronunciation errors but worse quality and vice versa.

#### CLI quick test (no prompt)
```
luxtts-mlx "Hello from MLX!" --out output.wav --device mlx
```

Use `--no-return-smooth` if you want the sharper 48k path.
Defaults are tuned for clarity: `--num-steps 5`, `--speed 0.92`, `--duration-pad-frames 16`.
Default output peak normalization is enabled: `--output-peak 0.92` (set `--output-peak 0` to disable).
If output feels too loud on speakers/headphones, use `--output-peak 0.8`.

#### CLI with prompt + optional prompt text
```
luxtts-mlx --text "Hello from MLX!" --prompt /path/to/prompt.wav --out output.wav --device mlx
luxtts-mlx --text "Hello from MLX!" --prompt /path/to/prompt.wav --prompt-text "Hello." --out output.wav --device mlx
```

Tip: providing `--prompt-text` skips Whisper prompt transcription load, which is faster and avoids extra multiprocessing warnings.
For long/noisy prompt files, use `--prompt-start` and `--ref-duration` to target a clean segment.
If prompt text is mis-transcribed or repeated, always pass explicit `--prompt-text`.

#### Optional fallback: torch vocoder with MLX diffusion
```
luxtts-mlx "Hello from MLX!" --prompt /path/to/prompt.wav --out output.wav --device mlx --vocoder torch
```

If you hit a Metal kernel error such as `Unable to load function four_step_mem_8192...`, re-run with `--vocoder torch`.

  
## Info

Q: How is this different from ZipVoice?

A: LuxTTS uses the same architecture but distilled to 4 steps with an improved sampling technique. It also uses a custom 48khz vocoder instead of the default 24khz version.

Q: Can it be even faster?

A: Yes, currently it uses float32. Float16 should be significantly faster(almost 2x).

## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [x] Release MPS support (thanks to @builtbybasit)
- [ ] Release code for float16 inference

## Acknowledgments

- [ZipVoice](https://github.com/k2-fsa/ZipVoice) for their excellent code and model.
- [Vocos](https://github.com/gemelo-ai/vocos.git) for their great vocoder.
  
## Final Notes

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma350@gmail.com
