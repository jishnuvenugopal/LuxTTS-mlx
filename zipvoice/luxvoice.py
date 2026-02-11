import os
import numpy as np
import torch

class LuxTTS:
    """
    LuxTTS class for encoding prompt and generating speech on cpu/cuda/mps.
    """

    def __init__(
        self,
        model_path='YatharthS/LuxTTS',
        device='cuda',
        threads=4,
        vocoder_backend='mlx',
        vocoder_device=None,
    ):
        if model_path == 'YatharthS/LuxTTS':
            model_path = None

        if device == 'mlx':
            try:
                os.environ.setdefault("LUXTTS_SUPPRESS_OPTIONAL_WARNINGS", "1")
                from zipvoice.mlx.modeling_utils import (
                    generate_mlx,
                    load_models_mlx,
                    load_transcriber_mlx,
                    process_audio_mlx,
                )
            except Exception as ex:
                raise ImportError("MLX backend not available. Install mlx and dependencies.") from ex
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_mlx(
                model_path,
                vocoder_backend=vocoder_backend,
                vocoder_device=vocoder_device,
                load_transcriber=False,
            )
            print("Loading model on MLX")

            self.model = model
            self.feature_extractor = feature_extractor
            self.vocos = vocos
            self.tokenizer = tokenizer
            self.transcriber = transcriber
            self.device = device
            self._generate_mlx = generate_mlx
            self._process_audio = process_audio_mlx
            self._load_transcriber = load_transcriber_mlx
            return

        from zipvoice.modeling_utils import (
            process_audio,
            generate,
            load_models_gpu,
            load_models_cpu,
        )

        # Auto-detect better device if cuda is requested but not available
        if device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available, switching to MPS")
                device = 'mps'
            else:
                print("CUDA not available, switching to CPU")
                device = 'cpu'

        if device == 'cpu':
            from zipvoice.onnx_modeling import generate_cpu
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path, threads)
            print("Loading model on CPU")
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(model_path, device=device)
            print("Loading model on GPU")

        self.model = model
        self.feature_extractor = feature_extractor
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber
        self.device = device
        self._process_audio = process_audio



    def encode_prompt(self, prompt_audio=None, duration=5, rms=0.001, prompt_text=None):
        """encodes audio prompt according to duration and rms(volume control)"""
        if self.device == "mlx" and prompt_audio is not None and not prompt_text and self.transcriber is None:
            self.transcriber = self._load_transcriber()
        device = "cpu" if self.device == "mlx" else self.device
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = self._process_audio(
            prompt_audio,
            self.transcriber,
            self.tokenizer,
            self.feature_extractor,
            device,
            target_rms=rms,
            duration=duration,
            prompt_text=prompt_text,
        )
        if self.device == "mlx":
            encode_dict = {
                "prompt_tokens": prompt_tokens,
                "prompt_features_lens": np.array(prompt_features_lens, dtype=np.int64),
                "prompt_features": np.array(prompt_features, dtype=np.float32),
                "prompt_rms": float(prompt_rms),
                "target_rms": float(rms),
            }
        else:
            encode_dict = {
                "prompt_tokens": prompt_tokens,
                "prompt_features_lens": prompt_features_lens,
                "prompt_features": prompt_features,
                "prompt_rms": prompt_rms,
                "target_rms": float(rms),
            }

        return encode_dict

    def generate_speech(
        self,
        text,
        encode_dict,
        num_steps=5,
        guidance_scale=3.0,
        t_shift=0.5,
        speed=0.92,
        return_smooth=True,
        duration_pad_frames=16,
    ):
        """encodes text and generates speech using flow matching model according to steps, guidance scale, and t_shift(like temp)"""

        prompt_tokens = encode_dict["prompt_tokens"]
        prompt_features_lens = encode_dict["prompt_features_lens"]
        prompt_features = encode_dict["prompt_features"]
        prompt_rms = encode_dict["prompt_rms"]
        target_rms = float(encode_dict.get("target_rms", 0.01))

        if hasattr(self.vocos, "return_48k"):
            if return_smooth == True:
                self.vocos.return_48k = False
            else:
                self.vocos.return_48k = True

        if self.device == 'mlx':
            final_wav = self._generate_mlx(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                self.model,
                self.vocos,
                self.tokenizer,
                num_step=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                target_rms=target_rms,
                duration_pad_frames=duration_pad_frames,
            )
        elif self.device == 'cpu':
            final_wav = generate_cpu(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                self.model,
                self.vocos,
                self.tokenizer,
                num_step=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                target_rms=target_rms,
            )
        else:
            final_wav = generate(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                self.model,
                self.vocos,
                self.tokenizer,
                num_step=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                target_rms=target_rms,
                duration_pad_frames=duration_pad_frames,
            )

        return final_wav if self.device == 'mlx' else final_wav.cpu()
