import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _logaddexp(a: mx.array, b: mx.array) -> mx.array:
    if hasattr(mx, "logaddexp"):
        return mx.logaddexp(a, b)
    # Stable logaddexp
    m = mx.maximum(a, b)
    return m + mx.log(mx.exp(a - m) + mx.exp(b - m))


def swoosh_l(x: mx.array) -> mx.array:
    zero = mx.zeros_like(x)
    return _logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def swoosh_r(x: mx.array) -> mx.array:
    zero = mx.zeros_like(x)
    return _logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class BiasNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = mx.array(log_scale)
        self.bias = mx.zeros((num_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = mx.expand_dims(bias, -1)
        diff = x - bias
        denom = mx.sqrt(mx.mean(diff * diff, axis=channel_dim, keepdims=True) + 1.0e-8)
        scale = mx.exp(self.log_scale) / denom
        return x * scale


class Balancer(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Whiten(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Dropout2(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        return x


class SwooshR(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return swoosh_r(x)


class ActivationDropoutAndLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        dropout_p: float = 0.0,
        dropout_shared_dim: Optional[int] = None,
        bias: bool = True,
        initial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim
        self.initial_scale = initial_scale

    def __call__(self, x: mx.array) -> mx.array:
        if self.activation == "SwooshL":
            x = swoosh_l(x)
        elif self.activation == "SwooshR":
            x = swoosh_r(x)
        return self.linear(x)


def ScaledLinear(*args, **kwargs) -> nn.Linear:
    return nn.Linear(*args, **kwargs)
