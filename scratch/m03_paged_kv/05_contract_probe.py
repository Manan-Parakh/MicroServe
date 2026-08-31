"""05 — The cache contract, probed by deliberately breaking it.

This is the M3 Phase 0 spike, kept because the findings ARE Phase 3's spec.
Four variants of a custom cache layer, run against StaticCache as reference:

  exact       return exactly what get_mask_sizes promised          -> identical
  padded      pad K/V to a BLOCK_SIZE multiple with loud finite
              garbage, AND declare the padded length               -> identical
  lying       return padded K/V but declare the exact length       -> RuntimeError
  poisoned    padded, declared, but the pad is inf instead of
              finite garbage                                       -> WRONG TOKENS

Takeaways:
  1. get_mask_sizes is a PROMISE made before update() runs. Breaking it
     crashes loudly -- which is the good failure mode.
  2. Block-granular padding is legal, so the gather never has to trim.
     Causality masks the tail: every padded slot is beyond every query.
  3. inf/NaN in a masked slot still poisons the output (0 * inf = NaN), and
     it does NOT raise. => allocate the pool with torch.zeros, never
     torch.empty. A reused block always holds finite KV, so free() need not
     zero anything.
  4. Layers must be PRE-CREATED. With layer_class_to_replicate, Cache
     short-circuits get_mask_sizes while the layer list is still empty --
     i.e. exactly on the prefill call. Run this file with PRECREATE = False
     to watch the padded variant break for that reason alone.

Verified on CPU + eager (the correctness gate's config, ADR-0010).
Re-verify under SDPA before quoting a GPU number from the paged path.

Run:  uv run python scratch/m03_paged_kv/05_contract_probe.py
"""

import sys

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, StaticCache

sys.path.insert(0, ".")
from engine.model_runner import ModelRunner  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 16
PRECREATE = True  # flip to False to see finding 4
MODE = "exact"


def pad_to_block(t, fill):
    n = t.shape[-2]
    padded = (n + BLOCK - 1) // BLOCK * BLOCK
    if padded == n:
        return t
    tail_shape = (*t.shape[:-2], padded - n, t.shape[-1])
    if fill == "inf":
        tail = torch.full(tail_shape, float("inf"), dtype=t.dtype, device=t.device)
    else:
        # Loud but finite: if this ever leaks into attention the tokens change.
        tail = torch.randn(tail_shape, dtype=t.dtype, device=t.device) * 100.0
    return torch.cat([t, tail], dim=-2)


class ProbeLayer(CacheLayerMixin):
    is_sliding = False

    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, k, v, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(k, v)
        self.keys = torch.cat([self.keys, k], dim=-2)
        self.values = torch.cat([self.values, v], dim=-2)
        if MODE == "exact":
            return self.keys, self.values
        fill = "inf" if MODE == "poisoned" else "finite"
        return pad_to_block(self.keys, fill), pad_to_block(self.values, fill)

    def get_mask_sizes(self, query_length):
        true_len = self.get_seq_length() + query_length
        if MODE in ("padded", "poisoned"):
            return (true_len + BLOCK - 1) // BLOCK * BLOCK, 0
        return true_len, 0  # "lying" declares the exact length while returning padded

    def get_seq_length(self):
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_length(self):
        return -1


class ProbeCache(Cache):
    def __init__(self, num_layers):
        if PRECREATE:
            super().__init__(layers=[ProbeLayer() for _ in range(num_layers)])
        else:
            super().__init__(layer_class_to_replicate=ProbeLayer)


def run(runner, lists, n_steps, factory):
    ids, mask = runner._left_pad(lists)
    pos = runner._position_ids_from_mask(mask)
    cache = factory(ids.shape[1] + n_steps)
    with torch.no_grad():
        out = runner.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                           past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1).tolist()
    gen = [[t] for t in nxt]
    for _ in range(n_steps - 1):
        nxt, mask = runner.decode_step(cache, nxt, mask)
        for row, t in zip(gen, nxt):
            row.append(t)
    return gen


def main():
    global MODE
    torch.manual_seed(0)
    runner = ModelRunner(MODEL_ID, device="cpu", attn_implementation="eager")
    num_layers = runner.model.config.num_hidden_layers
    lists = [
        runner.encode("The capital of France is"),
        runner.encode("Once upon a time, in a small village near the mountains, there"),
    ]

    ref = run(runner, lists, 12, lambda m: StaticCache(runner.model.config, max_cache_len=m))
    print(f"PRECREATE = {PRECREATE}")
    print(f"reference (StaticCache) row 0: {ref[0]}")
    print()
    for mode in ("exact", "padded", "lying", "poisoned"):
        MODE = mode
        try:
            got = run(runner, lists, 12, lambda m: ProbeCache(num_layers))
            verdict = "IDENTICAL" if got == ref else "WRONG TOKENS"
            print(f"[{mode:9}] {verdict:12} row 0: {got[0]}")
        except Exception as e:
            print(f"[{mode:9}] RAISED       {type(e).__name__}: {str(e)[:110]}")


if __name__ == "__main__":
    main()
