import functools
import time
import random
from typing import Sequence

import numpy as np
import tiktoken
import torch
from tqdm import tqdm
from implementations.karpathy import GPT as KarpathyGPT
from implementations.base import GPTConfig, BaseGPT
from implementations.memoized import MemoizedGPT

GPT = MemoizedGPT


def encode(text: str) -> list[int]:
    return tiktoken.get_encoding("gpt2").encode(text)


def decode(tokens: Sequence[int]) -> str:
    return tiktoken.get_encoding("gpt2").decode(tokens)


@functools.cache
def get_moby_dick_tokens() -> torch.Tensor:
    with open("data/moby_dick_no_newline.txt") as f:
        moby_txt = f.read()
    moby = np.array(encode(moby_txt))
    print("Moby Dick has {} tokens".format(len(moby)))
    return moby


def get_rand_input(batch_size, seq_len=1024):
    md_tokens = get_moby_dick_tokens()
    md_len = len(md_tokens)
    return np.array(
        [
            md_tokens[start_idx : start_idx + seq_len]
            for start_idx in (
                random.randint(0, md_len - seq_len) for _ in range(batch_size)
            )
        ]
    )


def sample(
    gpt: GPT,
    batch_size=1,
    prompt_tokens=512,
    get_rand_input=get_rand_input,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    u = get_rand_input(batch_size, prompt_tokens)

    cuda_u = torch.from_numpy(u).cuda()
    torch.cuda.synchronize()
    start.record()
    idx = gpt.generate(cuda_u, prompt_tokens)
    end.record()
    torch.cuda.synchronize()

    # visualize outputs
    idx = idx.to("cpu").numpy()
    for i in range(batch_size):
        print(decode(u[i]), "🩹", decode(idx[i][len(u[i]) :]))

    del cuda_u
    return start.elapsed_time(end) / 1e3


def benchmark(
    gpt, batch_size=1, prompt_tokens=256, sample_size=1, use_tqdm: bool = True
) -> float:
    samples = [
        sample(gpt, batch_size, prompt_tokens)
        for _ in tqdm(range(sample_size), disable=not use_tqdm)
    ]
    return samples


if __name__ == "__main__":
    gpt = MemoizedGPT.from_pretrained("gpt2-xl")
    gpt.eval()
    gpt.half()
    gpt.cuda()

    """
    tok = np.array([[691, 422, 511, 19501, 23755, 618, 287, 17087, 11, 475]])
    tok = torch.from_numpy(tok).cuda()
    logits, loss = gpt.forward(tok)
    print(logits[0, -1, :])

    tok = np.array([[476]])
    tok = torch.from_numpy(tok).cuda()
    logits, loss = gpt.forward(tok, offset=10)
    print(logits[0, -1, :])

    tok = np.array([[691, 422, 511, 19501, 23755, 618, 287, 17087, 11, 475, 476]])
    tok = torch.from_numpy(tok).cuda()
    logits, loss = gpt.forward(tok)
    print(logits[0, -1, :])

    exit()

    """

    for batch_size in tqdm([1]):
        print(f"batch_size: {batch_size}")
        print(benchmark(gpt, batch_size=batch_size)[0])
    exit()
