import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from implementations.base import BaseBlock, BaseGPT, BaseCausalSelfAttention, GPTConfig

MAX_LEN = 512
MAX_BATCH_SIZE = 100


class MemoizedCausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config):
        super().__init__(config=config)
        assert config.n_embd % config.n_head == 0

        # BEGIN CHANGES

        self.block_size = config.block_size
        self.cached_T = None
        self.register_buffer(
            "cached_k",
            torch.zeros(
                (
                    MAX_BATCH_SIZE,
                    config.n_head,
                    MAX_LEN or config.block_size,
                    config.n_embd // config.n_head,
                ),
                dtype=torch.float16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "cached_v",
            torch.zeros(
                (
                    MAX_BATCH_SIZE,
                    config.n_head,
                    config.block_size,
                    config.n_embd // config.n_head,
                ),
                dtype=torch.float16,
            ),
            persistent=False,
        )

        # END CHANGES

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        if T > 1:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side
            # output projection
            y = self.resid_dropout(self.c_proj(y))

            # initialize cache
            self.cached_k.zero_()
            self.cached_k[:B, :, :T, :] = k[:, :, :, :]
            self.cached_v.zero_()
            self.cached_v[:B, :, :T, :] = v[:, :, :, :]
            self.cached_T = T

            return y
        elif T == 1:
            # inference forward pass
            assert not self.training

            B, _, C = x.size()
            q_T, k_T, v_T = self.c_attn(x[:, 0, :]).split(
                self.n_embd, dim=1
            )  # (B, C) -> (B, 3C) -> 3 x (B, C)

            k_T = k_T.view(B, self.n_head, C // self.n_head)  # (B, nh, hs)
            q_T = q_T.view(B, self.n_head, C // self.n_head)  # (B, nh, hs)
            v_T = v_T.view(B, self.n_head, C // self.n_head)  # (B, nh, hs)

            self.cached_k[:B, :, self.cached_T, :] = k_T[:, :, :]
            self.cached_v[:B, :, self.cached_T, :] = v_T[:, :, :]

            k = self.cached_k[:B, :, : self.cached_T + 1, :]  # (B, nh, T, hs)
            v = self.cached_v[:B, :, : self.cached_T + 1, :]  # (B, nh, T, hs)

            self.cached_T = self.cached_T + 1

            # causal self-attention; Self-attend: (B, nh, 1, hs) x (B, nh, hs, T) -> (B, nh, 1, T)
            att = (q_T[:B, :, None, :] @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.size(-1))
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)  # don't need this lol

            y = att @ v  # (B, nh, 1, T) x (B, nh, T, hs) -> (B, nh, 1, hs)
            y = (
                y.transpose(1, 2).contiguous().view(B, 1, C)
            )  # re-assemble all head outputs side by side (B, C)
            y = self.c_proj(y)  # no dropout because inference only

            return y
        else:
            raise NotImplementedError


class MemoizedGPT(BaseGPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config, attn_class=MemoizedCausalSelfAttention)

    def forward(self, idx, targets=None, offset=0):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos += offset

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert max_new_tokens > 0

        T = idx.size(1)
        idx_cond = (
            idx
            if idx.size(1) <= self.config.block_size
            else idx[:, -self.config.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

        for pos in range(1, max_new_tokens):
            # T = 1 hits the incremental forward pass
            idx_incr = idx_next[:, :]
            logits, _ = self(idx_incr, offset=T + pos)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
