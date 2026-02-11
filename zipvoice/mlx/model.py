from typing import List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .solver import DistillEulerSolver, EulerSolver
from .utils import get_tokens_index, make_pad_mask, pad_labels, prepare_avg_tokens_durations
from .zipformer import TTSZipformer


class ZipVoiceMLX(nn.Module):
    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        self.solver = EulerSolver(self)

    def forward_fm_decoder(
        self,
        t: mx.array,
        xt: mx.array,
        text_condition: mx.array,
        speech_condition: mx.array,
        padding_mask: Optional[mx.array] = None,
        guidance_scale: Optional[mx.array] = None,
    ) -> mx.array:
        xt = mx.concatenate([xt, text_condition, speech_condition], axis=2)

        while t.ndim > 1 and t.shape[-1] == 1:
            t = mx.squeeze(t, axis=-1)
        if t.ndim == 0:
            t = mx.broadcast_to(t, (xt.shape[0],))

        if guidance_scale is not None:
            while guidance_scale.ndim > 1 and guidance_scale.shape[-1] == 1:
                guidance_scale = mx.squeeze(guidance_scale, axis=-1)
            if guidance_scale.ndim == 0:
                guidance_scale = mx.broadcast_to(guidance_scale, (xt.shape[0],))
            vt = self.fm_decoder(
                x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale
            )
        else:
            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask)
        return vt

    def forward_text_embed(self, tokens: List[List[int]]):
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id)
        embed = self.embed(tokens_padded)
        tokens_lens = mx.array([len(token) for token in tokens], dtype=mx.int64)
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])
        embed = self.text_encoder(x=embed, t=None, padding_mask=tokens_padding_mask)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: mx.array,
        tokens_lens: mx.array,
        features_lens: mx.array,
    ):
        num_frames = int(mx.max(features_lens).item())
        padding_mask = make_pad_mask(features_lens, max_len=num_frames)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)
        tokens_index = get_tokens_index(tokens_durations, num_frames)

        index = mx.expand_dims(tokens_index, axis=-1)
        index = mx.broadcast_to(index, (embed.shape[0], num_frames, embed.shape[-1]))
        text_condition = mx.take_along_axis(embed, index, axis=1)
        return text_condition, padding_mask

    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: mx.array,
        prompt_tokens: List[List[int]],
        prompt_features_lens: mx.array,
    ):
        tokens = [prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(tokens)
        return self.forward_text_condition(embed, tokens_lens, features_lens)

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: mx.array,
        speed: float,
        duration_pad_frames: int = 0,
    ):
        cat_tokens = [prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)]
        prompt_tokens_lens = mx.array([len(token) for token in prompt_tokens], dtype=mx.int64)
        tokens_lens = mx.array([len(token) for token in tokens], dtype=mx.int64)
        one = mx.ones_like(prompt_tokens_lens)
        prompt_tokens_lens = mx.maximum(prompt_tokens_lens, one)
        tokens_lens = mx.maximum(tokens_lens, one)

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)
        duration_pad_frames = max(int(duration_pad_frames), 0)
        features_lens = prompt_features_lens + mx.ceil(
            prompt_features_lens / prompt_tokens_lens * tokens_lens / speed
        ).astype(mx.int64)
        if duration_pad_frames > 0:
            features_lens = features_lens + duration_pad_frames

        return self.forward_text_condition(cat_embed, cat_tokens_lens, features_lens)

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: mx.array,
        prompt_features_lens: mx.array,
        features_lens: Optional[mx.array] = None,
        speed: float = 1.0,
        duration_pad_frames: int = 0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ):
        if duration not in ("real", "predict"):
            raise ValueError("duration must be 'real' or 'predict'")

        if duration == "predict":
            text_condition, padding_mask = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
                duration_pad_frames=duration_pad_frames,
            )
        else:
            if features_lens is None:
                raise ValueError("features_lens required when duration='real'")
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )

        batch_size, num_frames, _ = text_condition.shape
        speech_condition = mx.pad(
            prompt_features,
            ((0, 0), (0, num_frames - prompt_features.shape[1]), (0, 0)),
        )
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = mx.where(
            mx.expand_dims(speech_condition_mask, axis=-1),
            mx.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = mx.random.normal((batch_size, num_frames, prompt_features.shape[-1])).astype(mx.float32)

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )

        x1_wo_prompt_lens = mx.sum(mx.logical_not(padding_mask), axis=-1).astype(mx.int64) - prompt_features_lens
        max_wo_prompt = int(mx.max(x1_wo_prompt_lens).item())
        max_prompt = int(mx.max(prompt_features_lens).item())

        x1_np = np.array(x1)
        x1_prompt_np = np.zeros((batch_size, max_prompt, x1.shape[2]), dtype=np.float32)
        x1_wo_prompt_np = np.zeros((batch_size, max_wo_prompt, x1.shape[2]), dtype=np.float32)

        for i in range(batch_size):
            plen = int(prompt_features_lens[i].item())
            wlen = int(x1_wo_prompt_lens[i].item())
            x1_prompt_np[i, :plen] = x1_np[i, :plen]
            x1_wo_prompt_np[i, :wlen] = x1_np[i, plen : plen + wlen]

        x1_prompt = mx.array(x1_prompt_np)
        x1_wo_prompt = mx.array(x1_wo_prompt_np)
        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens


class ZipVoiceDistillMLX(ZipVoiceMLX):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        required_params = {
            "feat_dim",
            "fm_decoder_downsampling_factor",
            "fm_decoder_num_layers",
            "fm_decoder_cnn_module_kernel",
            "fm_decoder_dim",
            "fm_decoder_feedforward_dim",
            "fm_decoder_num_heads",
            "query_head_dim",
            "pos_head_dim",
            "value_head_dim",
            "pos_dim",
            "time_embed_dim",
        }
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        self.fm_decoder = TTSZipformer(
            in_dim=kwargs["feat_dim"] * 3,
            out_dim=kwargs["feat_dim"],
            downsampling_factor=kwargs["fm_decoder_downsampling_factor"],
            num_encoder_layers=kwargs["fm_decoder_num_layers"],
            cnn_module_kernel=kwargs["fm_decoder_cnn_module_kernel"],
            encoder_dim=kwargs["fm_decoder_dim"],
            feedforward_dim=kwargs["fm_decoder_feedforward_dim"],
            num_heads=kwargs["fm_decoder_num_heads"],
            query_head_dim=kwargs["query_head_dim"],
            pos_head_dim=kwargs["pos_head_dim"],
            value_head_dim=kwargs["value_head_dim"],
            pos_dim=kwargs["pos_dim"],
            use_time_embed=True,
            time_embed_dim=kwargs["time_embed_dim"],
            use_guidance_scale_embed=True,
            guidance_scale_embed_dim=kwargs["time_embed_dim"],
        )
        self.solver = DistillEulerSolver(self)
