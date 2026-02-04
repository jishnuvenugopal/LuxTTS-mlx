import mlx.core as mx
import mlx.nn as nn
import torch
import yaml

from vocos_mlx.model import ISTFTHead, VocosBackbone

from .weights import apply_state_dict_mlx


def _nonlinearity(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _linear_resample(audio: mx.array, src_sr: int, dst_sr: int) -> mx.array:
    if src_sr == dst_sr:
        return audio
    length = audio.shape[-1]
    new_length = int(round(length * float(dst_sr) / float(src_sr)))
    pos = mx.linspace(0, length - 1, new_length)
    idx0 = mx.astype(mx.floor(pos), mx.int64)
    idx1 = mx.minimum(idx0 + 1, length - 1)
    w = pos - idx0
    while w.ndim < audio.ndim:
        w = mx.expand_dims(w, axis=0)
    a0 = mx.take(audio, idx0, axis=-1)
    a1 = mx.take(audio, idx1, axis=-1)
    return (1.0 - w) * a0 + w * a1


def _crossover_merge_linkwitz_riley(
    path1_48k: mx.array,
    path2_48k: mx.array,
    sample_rate: int = 48000,
    cutoff: int = 4000,
    transition_bins: int = 8,
) -> mx.array:
    spec1 = mx.fft.rfft(path1_48k)
    spec2 = mx.fft.rfft(path2_48k)

    n_bins = spec1.shape[-1]
    cutoff_bin = int((cutoff / (sample_rate / 2)) * n_bins)

    mask = mx.ones((n_bins,), dtype=spec1.dtype)
    half = transition_bins // 2
    start = max(0, cutoff_bin - half)
    end = min(n_bins, cutoff_bin + half)
    actual_width = end - start

    if actual_width > 0:
        x = mx.linspace(-1, 1, actual_width)
        fade = 3 * mx.power((x + 1) / 2, 2) - 2 * mx.power((x + 1) / 2, 3)
        mask = mx.concatenate(
            [
                mx.zeros((start,), dtype=mask.dtype),
                fade,
                mx.ones((n_bins - end,), dtype=mask.dtype),
            ]
        )

    merged_spec = (spec1 * mask) + (spec2 * (1.0 - mask))
    return mx.fft.irfft(merged_spec, n=path1_48k.shape[-1])


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.ones((1, 1, channels), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x + (1.0 / (self.alpha + 1.0e-9)) * mx.sin(self.alpha * x) ** 2


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        self.snake1 = Snake1d(in_channels)
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None

        self.norm2 = nn.GroupNorm(32, self.out_channels, eps=1e-6, pytorch_compatible=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.snake2 = Snake1d(self.out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
                self.nin_shortcut = None
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
                self.conv_shortcut = None
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None

    def __call__(self, x: mx.array, temb: mx.array | None = None) -> mx.array:
        h = self.norm1(x)
        h = self.snake1(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(_nonlinearity(temb))

        h = self.norm2(h)
        h = self.snake2(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            elif self.nin_shortcut is not None:
                x = self.nin_shortcut(x)

        return x + h


class UpSamplerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_factors: list[int],
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors or [])
        self.kernel_sizes = list(kernel_sizes or [8] * len(self.upsample_factors))

        if len(self.kernel_sizes) != len(self.upsample_factors):
            raise ValueError("kernel_sizes and upsample_factors must have the same length")

        self.upsample_layers = []
        self.resnet_blocks = []
        self.out_proj = nn.Linear(
            self.in_channels // (2 ** len(self.upsample_factors)),
            self.in_channels,
            bias=True,
        )
        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors)):
            c_in = self.in_channels // (2 ** i)
            c_out = self.in_channels // (2 ** (i + 1))
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    c_in,
                    c_out,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            self.resnet_blocks.append(
                ResnetBlock(in_channels=c_out, out_channels=c_out, dropout=0.0, temb_channels=0)
            )
        self.final_snake = Snake1d(self.in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        for up, rsblk in zip(self.upsample_layers, self.resnet_blocks):
            x = rsblk(up(x))
        x = self.out_proj(x)
        return self.final_snake(x)


class LuxVocoderMLX(nn.Module):
    def __init__(
        self,
        backbone: VocosBackbone,
        head: ISTFTHead,
        upsampler: UpSamplerBlock,
        head_48k: ISTFTHead,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.upsampler = upsampler
        self.head_48k = head_48k
        self.sample_rate = sample_rate
        self.freq_range = 4000
        self.return_48k = True

    def decode(self, features_input: mx.array) -> mx.array:
        features = self.backbone(features_input)
        upsampled = self.upsampler(features)
        pred_audio = self.head_48k(upsampled)
        pred_audio2 = self.head(features)
        pred_audio2 = _linear_resample(pred_audio2, self.sample_rate, self.sample_rate * 2)
        min_len = min(pred_audio.shape[-1], pred_audio2.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        pred_audio2 = pred_audio2[..., :min_len]
        merged = _crossover_merge_linkwitz_riley(
            pred_audio, pred_audio2, sample_rate=self.sample_rate * 2, cutoff=self.freq_range
        )
        return merged if self.return_48k else pred_audio2


def load_vocoder_mlx(model_path: str) -> LuxVocoderMLX:
    config_path = f"{model_path}/vocoder/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone_args = config["backbone"]["init_args"]
    head_args = config["head"]["init_args"]
    head_48k_args = config["head_48k"]["init_args"]
    upsampler_args = config["upsampler"]["init_args"]

    sample_rate = config.get("feature_extractor", {}).get("init_args", {}).get("sample_rate", 24000)

    vocoder = LuxVocoderMLX(
        backbone=VocosBackbone(**backbone_args),
        head=ISTFTHead(**head_args),
        upsampler=UpSamplerBlock(**upsampler_args),
        head_48k=ISTFTHead(**head_48k_args),
        sample_rate=sample_rate,
    )

    state_dict = torch.load(f"{model_path}/vocoder/vocos.bin", map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    apply_state_dict_mlx(vocoder, state_dict)
    return vocoder
