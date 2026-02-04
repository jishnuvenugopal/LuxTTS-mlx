import math
import copy
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    Dropout2,
    Identity,
    ScaledLinear,
    SwooshR,
    Whiten,
    swoosh_l,
    swoosh_r,
)


def _arctan(x: mx.array) -> mx.array:
    if hasattr(mx, "arctan"):
        return mx.arctan(x)
    return mx.atan(x)


def timestep_embedding(timesteps: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(
        -math.log(max_period)
        * mx.arange(0, half, dtype=mx.float32)
        / half
    )

    if timesteps.ndim == 2:
        timesteps = mx.transpose(timesteps, (1, 0))

    args = timesteps[..., None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[..., :1])], axis=-1)
    return emb


class ModuleList(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = list(modules)

    def __iter__(self):
        return iter(self.modules)

    def __len__(self):
        return len(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]


class Sequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.modules:
            x = module(x)
        return x

    def __getitem__(self, idx):
        return self.modules[idx]


class BypassModule(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.bypass_scale = mx.full((embed_dim,), 0.5, dtype=mx.float32)

    def __call__(self, src_orig: mx.array, src: mx.array) -> mx.array:
        return src_orig + (src - src_orig) * self.bypass_scale


class SimpleDownsample(nn.Module):
    def __init__(self, downsample: int):
        super().__init__()
        self.bias = mx.zeros((downsample,), dtype=mx.float32)
        self.downsample = downsample

    def __call__(self, src: mx.array) -> mx.array:
        seq_len, batch_size, in_channels = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds
        pad = d_seq_len * ds - seq_len
        if pad > 0:
            last = mx.expand_dims(src[-1], axis=0)
            src_extra = mx.broadcast_to(last, (pad, batch_size, in_channels))
            src = mx.concatenate([src, src_extra], axis=0)
        src = mx.reshape(src, (d_seq_len, ds, batch_size, in_channels))
        weights = mx.softmax(self.bias, axis=0)
        weights = mx.reshape(weights, (1, ds, 1, 1))
        return mx.sum(src * weights, axis=1)


class SimpleUpsample(nn.Module):
    def __init__(self, upsample: int):
        super().__init__()
        self.upsample = upsample

    def __call__(self, src: mx.array) -> mx.array:
        seq_len, batch_size, num_channels = src.shape
        up = self.upsample
        src = mx.expand_dims(src, axis=1)
        src = mx.broadcast_to(src, (seq_len, up, batch_size, num_channels))
        return mx.reshape(src, (seq_len * up, batch_size, num_channels))


class CompactRelPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.length_factor = length_factor
        self.max_len = max_len

    def __call__(self, x: mx.array, left_context_len: int = 0) -> mx.array:
        T = x.shape[0] + left_context_len
        positions = mx.arange(-(T - 1), T, dtype=mx.float32)[:, None]

        freqs = 1 + mx.arange(self.embed_dim // 2, dtype=mx.float32)
        compression_length = self.embed_dim ** 0.5
        x_compressed = (
            compression_length
            * mx.sign(positions)
            * (mx.log(mx.abs(positions) + compression_length) - math.log(compression_length))
        )
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        x_atan = _arctan(x_compressed / length_scale)

        cosines = mx.cos(x_atan * freqs)
        sines = mx.sin(x_atan * freqs)

        pe = mx.concatenate([cosines, sines], axis=1)
        pe = mx.concatenate(
            [pe[:, :-1], mx.ones((pe.shape[0], 1), dtype=pe.dtype)], axis=1
        )

        center = pe.shape[0] // 2
        x_size_left = x.shape[0] + left_context_len
        start = center - x_size_left + 1
        end = center + x.shape[0]
        pos_emb = pe[start:end]
        pos_emb = mx.expand_dims(pos_emb, axis=0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True)
        self.linear_pos = ScaledLinear(pos_dim, num_heads * pos_head_dim, bias=False)
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        key_padding_mask: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.in_proj(x)
        seq_len, batch_size, _ = x.shape
        query_dim = self.query_head_dim * self.num_heads

        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        p = x[..., 2 * query_dim :]

        q = self.copy_query(q)
        p = self.copy_pos_query(p)

        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.query_head_dim))

        q = mx.transpose(q, (2, 1, 0, 3))
        p = mx.transpose(p, (2, 1, 0, 3))
        k = mx.transpose(k, (2, 1, 3, 0))

        attn_scores = mx.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1
        pos_emb = mx.reshape(pos_emb, (-1, seq_len2, self.num_heads, self.pos_head_dim))
        pos_emb = mx.transpose(pos_emb, (2, 0, 3, 1))

        pos_scores = mx.matmul(p, pos_emb)
        pos_scores = _relative_to_absolute(pos_scores, seq_len)
        attn_scores = attn_scores + pos_scores

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                mask = mx.expand_dims(mx.expand_dims(attn_mask, axis=0), axis=0)
            else:
                mask = mx.expand_dims(attn_mask, axis=0)
            attn_scores = mx.where(mask, -1000.0, attn_scores)

        if key_padding_mask is not None:
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, axis=0), axis=2)
            attn_scores = mx.where(mask, -1000.0, attn_scores)

        attn_weights = mx.softmax(attn_scores, axis=-1)
        return attn_weights


def _relative_to_absolute(pos_scores: mx.array, seq_len: int) -> mx.array:
    num_heads, batch_size, time1, n = pos_scores.shape
    rows = mx.arange(time1 - 1, -1, -1)
    cols = mx.arange(seq_len)
    idx = mx.astype(rows[:, None] + cols[None, :], mx.int64)
    idx = mx.expand_dims(idx, axis=0)
    idx = mx.broadcast_to(idx, (num_heads * batch_size, time1, seq_len))
    pos_scores_flat = mx.reshape(pos_scores, (num_heads * batch_size, time1, n))
    gathered = mx.take_along_axis(pos_scores_flat, idx, axis=2)
    return mx.reshape(gathered, (num_heads, batch_size, time1, seq_len))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.out_proj = ScaledLinear(num_heads * value_head_dim, embed_dim, bias=True)
        self.whiten = Whiten()

    def __call__(self, x: mx.array, attn_weights: mx.array) -> mx.array:
        seq_len, batch_size, _ = x.shape
        num_heads = attn_weights.shape[0]
        x = self.in_proj(x)
        x = mx.reshape(x, (seq_len, batch_size, num_heads, -1))
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.matmul(attn_weights, x)
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.reshape(x, (seq_len, batch_size, -1))
        x = self.out_proj(x)
        x = self.whiten(x)
        return x


class FeedforwardModule(nn.Module):
    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
        )
        self.out_whiten = Whiten()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.in_proj(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.tanh = nn.Tanh()
        self.out_proj = ScaledLinear(hidden_channels, channels, bias=True)
        self.whiten1 = Whiten()
        self.whiten2 = Whiten()

    def __call__(self, x: mx.array, attn_weights: mx.array) -> mx.array:
        x = self.in_proj(x)
        seq_len, batch_size, _ = x.shape
        chunk = x.shape[2] // 3
        x_all = x
        s = x_all[:, :, :chunk]
        x = x_all[:, :, chunk : 2 * chunk]
        y = x_all[:, :, 2 * chunk :]
        s = self.tanh(s)
        s = mx.reshape(s, (seq_len, batch_size, self.hidden_channels))
        x = self.whiten1(x)
        x = x * s

        num_heads = attn_weights.shape[0]
        x = mx.reshape(x, (seq_len, batch_size, num_heads, -1))
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.matmul(attn_weights, x)
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.reshape(x, (seq_len, batch_size, -1))

        x = x * y
        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        bottleneck_dim = channels
        self.in_proj = nn.Linear(channels, 2 * bottleneck_dim)
        self.sigmoid = nn.Sigmoid()
        self.depthwise_conv = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck_dim,
            bias=True,
        )
        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            bias=True,
        )
        self.whiten = Whiten()

    def __call__(self, x: mx.array, src_key_padding_mask: Optional[mx.array] = None) -> mx.array:
        x = self.in_proj(x)
        chunk = x.shape[2] // 2
        x, s = x[:, :, :chunk], x[:, :, chunk:]
        s = self.sigmoid(s)
        x = x * s

        x = mx.transpose(x, (1, 0, 2))  # (batch, time, channels)
        if src_key_padding_mask is not None:
            mask = mx.expand_dims(src_key_padding_mask, axis=-1)
            x = mx.where(mask, 0.0, x)
        x = self.depthwise_conv(x)
        x = mx.transpose(x, (1, 0, 2))
        x = self.whiten(x)
        x = self.out_proj(x)
        return x


class Zipformer2EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: float,
        cnn_module_kernel: int,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bypass = BypassModule(embed_dim)
        self.bypass_mid = BypassModule(embed_dim)
        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)
        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)
        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)
        self.use_conv = use_conv
        if self.use_conv:
            self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel)
            self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel)
        self.norm = BiasNorm(embed_dim)

    def __call__(
        self,
        src: mx.array,
        pos_emb: mx.array,
        time_emb: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        src_orig = src
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )
        if time_emb is not None:
            src = src + time_emb
        src = src + self.feed_forward1(src)

        selected_attn_weights = attn_weights[:1]
        na = self.nonlin_attention(src, selected_attn_weights)
        src = src + na

        src = src + self.self_attn1(src, attn_weights)

        if self.use_conv:
            if time_emb is not None:
                src = src + time_emb
            src = src + self.conv_module1(src, src_key_padding_mask=src_key_padding_mask)

        src = src + self.feed_forward2(src)
        src = self.bypass_mid(src_orig, src)

        src = src + self.self_attn2(src, attn_weights)

        if self.use_conv:
            if time_emb is not None:
                src = src + time_emb
            src = src + self.conv_module2(src, src_key_padding_mask=src_key_padding_mask)

        src = src + self.feed_forward3(src)
        src = self.norm(src)
        src = self.bypass(src_orig, src)
        return src


class Zipformer2Encoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        embed_dim: int,
        time_embed_dim: int,
        pos_dim: int,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.0, length_factor=1.0)
        if time_embed_dim != -1:
            self.time_emb = Sequential(SwooshR(), nn.Linear(time_embed_dim, embed_dim))
        else:
            self.time_emb = None
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def __call__(
        self,
        src: mx.array,
        time_emb: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        pos_emb = self.encoder_pos(src)
        if self.time_emb is not None and time_emb is not None:
            time_emb = self.time_emb(time_emb)
        else:
            time_emb = None
        output = src
        for layer in self.layers:
            output = layer(
                output,
                pos_emb,
                time_emb=time_emb,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output


class DownsampledZipformer2Encoder(nn.Module):
    def __init__(self, encoder: nn.Module, dim: int, downsample: int):
        super().__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(downsample)
        self.out_combiner = BypassModule(dim)

    def __call__(
        self,
        src: mx.array,
        time_emb: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if time_emb is not None and time_emb.ndim == 3:
            time_emb = time_emb[::ds]
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[..., ::ds]

        src = self.encoder(
            src,
            time_emb=time_emb,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        src = src[: src_orig.shape[0]]
        return self.out_combiner(src_orig, src)


class TTSZipformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsampling_factor: Union[int, Tuple[int]] = (2, 4),
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        encoder_dim: int = 384,
        query_head_dim: int = 24,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        num_heads: int = 8,
        feedforward_dim: int = 1536,
        pos_dim: int = 192,
        dropout: float = 0.0,
        use_time_embed: bool = True,
        time_embed_dim: int = 192,
        use_guidance_scale_embed: bool = False,
        guidance_scale_embed_dim: int = 192,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,)

        def _to_tuple(x):
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            return x

        self.downsampling_factor = downsampling_factor
        num_encoder_layers = _to_tuple(num_encoder_layers)
        cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.use_time_embed = use_time_embed
        self.use_guidance_scale_embed = use_guidance_scale_embed
        self.time_embed_dim = time_embed_dim
        self.guidance_scale_embed_dim = guidance_scale_embed_dim

        self.in_proj = nn.Linear(in_dim, encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, out_dim)

        encoders = []
        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim,
                pos_dim=pos_dim,
                num_heads=num_heads,
                query_head_dim=query_head_dim,
                pos_head_dim=pos_head_dim,
                value_head_dim=value_head_dim,
                feedforward_dim=feedforward_dim,
                use_conv=use_conv,
                cnn_module_kernel=cnn_module_kernel[i],
                dropout=dropout,
            )
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                embed_dim=encoder_dim,
                time_embed_dim=time_embed_dim if self.use_time_embed else -1,
                pos_dim=pos_dim,
            )
            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim,
                    downsample=downsampling_factor[i],
                )
            encoders.append(encoder)
        self.encoders = ModuleList(encoders)

        if self.use_time_embed:
            self.time_embed = Sequential(
                nn.Linear(time_embed_dim, time_embed_dim * 2),
                SwooshR(),
                nn.Linear(time_embed_dim * 2, time_embed_dim),
            )
        else:
            self.time_embed = None

        if self.use_guidance_scale_embed:
            self.guidance_scale_embed = ScaledLinear(
                guidance_scale_embed_dim,
                time_embed_dim,
                bias=False,
            )
        else:
            self.guidance_scale_embed = None

    def __call__(
        self,
        x: mx.array,
        t: Optional[mx.array] = None,
        padding_mask: Optional[mx.array] = None,
        guidance_scale: Optional[mx.array] = None,
    ) -> mx.array:
        x = mx.transpose(x, (1, 0, 2))
        x = self.in_proj(x)

        if t is not None:
            time_emb = timestep_embedding(t, self.time_embed_dim)
            if guidance_scale is not None and self.guidance_scale_embed is not None:
                guidance_scale_emb = self.guidance_scale_embed(
                    timestep_embedding(guidance_scale, self.guidance_scale_embed_dim)
                )
                time_emb = time_emb + guidance_scale_emb
            if self.time_embed is not None:
                time_emb = self.time_embed(time_emb)
        else:
            time_emb = None

        attn_mask = None
        for module in self.encoders:
            x = module(
                x,
                time_emb=time_emb,
                src_key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )
        x = self.out_proj(x)
        x = mx.transpose(x, (1, 0, 2))
        return x
