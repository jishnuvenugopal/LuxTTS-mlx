from __future__ import annotations

import sys

import numpy as np
import torch


_PATCHED_LINKWITZ = False


def _crossover_merge_linkwitz_riley_safe(
    path1_48k: torch.Tensor,
    path2_48k: torch.Tensor,
    sample_rate: int = 48000,
    cutoff: float = 4000.0,
    transition_bins: int = 8,
) -> torch.Tensor:
    a = path1_48k.detach().to(torch.float32).cpu().numpy()
    b = path2_48k.detach().to(torch.float32).cpu().numpy()

    spec1 = np.fft.rfft(a, axis=-1)
    spec2 = np.fft.rfft(b, axis=-1)

    n_bins = int(spec1.shape[-1])
    cutoff_bin = int((float(cutoff) / (float(sample_rate) / 2.0)) * n_bins)
    cutoff_bin = max(0, min(cutoff_bin, n_bins - 1))

    mask = np.ones((n_bins,), dtype=np.float32)
    half = max(1, int(transition_bins) // 2)
    start = max(0, cutoff_bin - half)
    end = min(n_bins, cutoff_bin + half)
    width = end - start

    if width > 0:
        x = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
        t = (x + 1.0) / 2.0
        fade = (3.0 * np.square(t)) - (2.0 * np.power(t, 3))
        mask[:start] = 0.0
        mask[start:end] = fade
        mask[end:] = 1.0
    else:
        mask[:cutoff_bin] = 0.0
        mask[cutoff_bin:] = 1.0

    merged_spec = (spec1 * mask) + (spec2 * (1.0 - mask))
    merged = np.fft.irfft(merged_spec, n=a.shape[-1], axis=-1).astype(np.float32, copy=False)
    merged_t = torch.from_numpy(np.ascontiguousarray(merged))
    return merged_t.to(device=path1_48k.device, dtype=path1_48k.dtype)


def apply_linacodec_linkwitz_patch() -> None:
    global _PATCHED_LINKWITZ
    if _PATCHED_LINKWITZ:
        return

    try:
        import linacodec.vocoder.linkwitz as linkwitz_mod
        import linacodec.vocoder.vocos as vocos_mod
    except Exception:
        return

    linkwitz_mod.crossover_merge_linkwitz_riley = _crossover_merge_linkwitz_riley_safe
    vocos_mod.crossover_merge_linkwitz_riley = _crossover_merge_linkwitz_riley_safe

    # Patch any already-imported linacodec vocoder modules that may have copied
    # the symbol via "from ... import crossover_merge_linkwitz_riley".
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("linacodec.vocoder"):
            continue
        if hasattr(module, "crossover_merge_linkwitz_riley"):
            setattr(module, "crossover_merge_linkwitz_riley", _crossover_merge_linkwitz_riley_safe)

    _PATCHED_LINKWITZ = True
