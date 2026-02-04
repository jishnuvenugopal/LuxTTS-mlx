from typing import List

import numpy as np
import mlx.core as mx


def pad_labels(labels: List[List[int]], pad_id: int) -> mx.array:
    labels = [token_ids + [pad_id] for token_ids in labels]
    max_len = max(len(token_ids) for token_ids in labels)
    padded = [token_ids + [pad_id] * (max_len - len(token_ids)) for token_ids in labels]
    return mx.array(np.array(padded, dtype=np.int64))


def prepare_avg_tokens_durations(features_lens, tokens_lens):
    features_lens = np.array(features_lens, dtype=np.int64)
    tokens_lens = np.array(tokens_lens, dtype=np.int64)
    durations = []
    for i in range(len(features_lens)):
        utt_duration = int(features_lens[i])
        tok_len = max(int(tokens_lens[i]), 1)
        avg_token_duration = max(int(utt_duration // tok_len), 1)
        durations.append([avg_token_duration] * tok_len)
    return durations


def get_tokens_index(durations: List[List[int]], num_frames: int) -> mx.array:
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = np.zeros((batch_size, num_frames), dtype=np.int64)
    for b in range(batch_size):
        cur_frame = 0
        for i, d in enumerate(durations[b]):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        if cur_frame != num_frames:
            raise ValueError(f"Duration mismatch: {cur_frame} != {num_frames}")
    return mx.array(ans)


def make_pad_mask(lengths, max_len: int = 0) -> mx.array:
    lengths = np.array(lengths, dtype=np.int64)
    if lengths.ndim != 1:
        raise ValueError("lengths must be 1-D")
    max_len = max(max_len, int(lengths.max()))
    seq_range = np.arange(max_len, dtype=np.int64)
    mask = seq_range[None, :] >= lengths[:, None]
    return mx.array(mask)
