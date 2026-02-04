from typing import Any, Dict, Tuple

import numpy as np
import mlx.core as mx


def _compute_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    if weight_g.ndim == 1:
        shape = (weight_g.shape[0],) + (1,) * (weight_v.ndim - 1)
        weight_g = weight_g.reshape(shape)
    norm_axes = tuple(range(1, weight_v.ndim))
    v_norm = np.linalg.norm(weight_v, axis=norm_axes, keepdims=True)
    return weight_v * (weight_g / (v_norm + 1.0e-12))


def _inject_weight_norm(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_group(base: str) -> Dict[str, np.ndarray]:
        group = groups.get(base)
        if group is None:
            group = {}
            groups[base] = group
        return group

    for name, tensor in state_dict.items():
        if name.endswith(".parametrizations.weight.original0"):
            base = name[: -len(".parametrizations.weight.original0")]
            _get_group(base)["g"] = tensor.detach().cpu().numpy()
        elif name.endswith(".parametrizations.weight.original1"):
            base = name[: -len(".parametrizations.weight.original1")]
            _get_group(base)["v"] = tensor.detach().cpu().numpy()
        elif name.endswith(".weight_g"):
            base = name[: -len(".weight_g")]
            _get_group(base)["g"] = tensor.detach().cpu().numpy()
        elif name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            _get_group(base)["v"] = tensor.detach().cpu().numpy()

    if not groups:
        return state_dict

    merged = dict(state_dict)
    for base, parts in groups.items():
        if "g" not in parts or "v" not in parts:
            continue
        merged[base + ".weight"] = _compute_weight_norm(parts["g"], parts["v"])
    return merged


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
    state_dict = _inject_weight_norm(state_dict)
    for name, tensor in state_dict.items():
        if isinstance(tensor, np.ndarray):
            value = tensor
        else:
            value = tensor.detach().cpu().numpy()
        try:
            _set_param(model, name, value)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
