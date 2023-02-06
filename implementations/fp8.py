from implementations.base import GPTConfig
from implementations.memoized import MemoizedGPT, MemoizedCausalSelfAttention

import torch
import torch.nn as nn


class FP8MemoizedGPT(MemoizedGPT):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        gpt = super().from_pretrained(*args, **kwargs)
        return gpt
