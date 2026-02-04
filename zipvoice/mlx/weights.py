from typing import Any

import numpy as np
import mlx.core as mx


def _set_param(obj: Any, name: str, value: np.ndarray) -> None:
    parts = name.split(".")
    cur = obj
    for part in parts[:-1]:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    leaf = parts[-1]

    if leaf == "weight" and cur.__class__.__name__ == "Conv1d":
        if value.ndim == 3:
            value = np.transpose(value, (0, 2, 1))
    if leaf == "weight" and cur.__class__.__name__ == "ConvTranspose1d":
        if value.ndim == 3:
            value = np.transpose(value, (1, 2, 0))
    if leaf == "alpha" and value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 1:
        value = np.transpose(value, (0, 2, 1))

    setattr(cur, leaf, mx.array(value))


def apply_state_dict_mlx(model: Any, state_dict) -> None:
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().numpy()
        try:
            _set_param(model, name, value)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
