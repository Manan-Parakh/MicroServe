"""02 — What shapes actually cross the cache boundary?

Instruments a real forward pass with a cache layer that only *watches*. Shows
the exact tensors the model hands to `update()` and what it expects back, for
both the prefill call and a decode step.

The one rule to take away:
    get_mask_sizes(query_length) -> (kv_length, kv_offset)
is a PROMISE, made before update() runs, about how wide the K/V you return
will be. Break it and you get a RuntimeError (see 05_contract_probe.py).

Run:  uv run python scratch/m03_paged_kv/02_kv_shapes.py
"""

import sys

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

sys.path.insert(0, ".")
from engine.model_runner import ModelRunner  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOG = []
PHASE = "prefill"


class WatchingLayer(CacheLayerMixin):
    """Stores contiguously (this is DynamicLayer). It only adds printouts."""

    is_sliding = False

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, k, v, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(k, v)
        before = self.get_seq_length()
        self.keys = torch.cat([self.keys, k], dim=-2)
        self.values = torch.cat([self.values, v], dim=-2)
        if self.layer_idx == 0:
            LOG.append(
                f"  [{PHASE}] layer 0 update(): incoming k {tuple(k.shape)} "
                f"| cache had {before} -> returns {tuple(self.keys.shape)}"
            )
        return self.keys, self.values

    def get_mask_sizes(self, query_length):
        sizes = (self.get_seq_length() + query_length, 0)
        if self.layer_idx == 0:
            LOG.append(f"  [{PHASE}] layer 0 get_mask_sizes(q={query_length}) -> {sizes}")
        return sizes

    def get_seq_length(self):
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_length(self):
        return -1


class WatchingCache(Cache):
    def __init__(self, num_layers):
        # Pre-created, NOT layer_class_to_replicate -- see 05_contract_probe.py
        super().__init__(layers=[WatchingLayer(i) for i in range(num_layers)])


def main():
    global PHASE
    runner = ModelRunner(MODEL_ID, device="cpu", attn_implementation="eager")
    num_layers = runner.model.config.num_hidden_layers

    prompts = [
        "The capital of France is",
        "Once upon a time, in a small village near the mountains, there",
    ]
    lists = [runner.encode(p) for p in prompts]
    print(lists)
    print("prompt token counts:", [len(x) for x in lists])
    print()

    ids, mask = runner._left_pad(lists)
    print(f"ids: {ids}")
    print(f"attn_mask: {mask}")
    pos = runner._position_ids_from_mask(mask)
    print("--- batching: left-padding ------------------------------------------")
    print(f"  input_ids       {tuple(ids.shape)}   [batch, padded_prompt_len]")
    print(f"  attention_mask  {tuple(mask.shape)}   1 = real token, 0 = left pad")
    print(f"    row 0: {mask[0].tolist()}")
    print(f"    row 1: {mask[1].tolist()}")
    print(f"  position_ids row 0: {pos[0].tolist()}")
    print("    (pad columns get placeholder position 1; they are masked out anyway)")
    print()

    cache = WatchingCache(num_layers)
    with torch.no_grad():
        out = runner.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                           past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1).tolist()

    print("--- PREFILL: one forward over the whole prompt ----------------------")
    for line in LOG:
        print(line)
    print(f"  logits {tuple(out.logits.shape)}  [batch, q_len, vocab]; we read [:, -1, :]")
    print()
    LOG.clear()

    PHASE = "decode"
    for _ in range(2):
        nxt, mask = runner.decode_step(cache, nxt, mask)
    print("--- DECODE: one forward per step, q_len == 1 ------------------------")
    for line in LOG:
        print(line)
    print(f"  attention_mask has grown to {tuple(mask.shape)}")
    print()

    k = cache.layers[0].keys
    print("--- the cached K tensor, layer 0 ------------------------------------")
    print(f"  shape {tuple(k.shape)}  = [batch, num_kv_heads, seq_len, head_dim]")
    print("  THIS is the layout your paged gather must reproduce exactly.")
    print(f"  Note dim 1 is {k.shape[1]} (KV heads), not 14 (query heads).")
    print()
    print("  Contiguous storage means row 0 and row 1 sit in one rectangle of")
    print(f"  width {k.shape[2]}, even though row 0 only has "
          f"{int(mask[0].sum())} real tokens. That waste is what paging removes.")


if __name__ == "__main__":
    main()
