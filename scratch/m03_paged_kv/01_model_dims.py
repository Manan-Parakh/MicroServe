"""01 — Where do the numbers come from?

Everything in M3 is arithmetic on six numbers pulled off the model config.
Get these wrong and the pool is the wrong size; get `num_key_value_heads`
wrong and it is wrong by 7x on this model.

Run:  uv run python scratch/m03_paged_kv/01_model_dims.py
"""

from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16


def main():
    cfg = AutoConfig.from_pretrained(MODEL_ID)

    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    hidden = cfg.hidden_size
    # Qwen2.5 has no explicit `head_dim`; it is hidden_size / num_attention_heads.
    head_dim = getattr(cfg, "head_dim", None) or hidden // num_q_heads

    print(f"model                 {MODEL_ID}")
    print(f"num_hidden_layers     {num_layers}")
    print(f"num_attention_heads   {num_q_heads}   <- QUERY heads")
    print(f"num_key_value_heads   {num_kv_heads}   <- KEY/VALUE heads (what the cache stores)")
    print(f"hidden_size           {hidden}")
    print(f"head_dim              {head_dim}  (= hidden_size / num_attention_heads)")
    print()

    print("--- GQA: why the two head counts differ ------------------------------")
    print(f"  {num_q_heads} query heads share {num_kv_heads} key/value heads")
    print(f"  -> each KV head is reused by {num_q_heads // num_kv_heads} query heads")
    print("  The KV cache only ever stores KEY/VALUE heads. Query heads are")
    print("  recomputed every forward and never cached.")
    print("  Sizing the pool with num_attention_heads would overshoot by")
    print(f"  {num_q_heads / num_kv_heads:.0f}x -- the roadmap's stated pool shape says")
    print("  `num_heads`; it must be `num_key_value_heads`.")
    print()

    # The model as ModelRunner actually loads it, so dtype is the real one.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    dtype = next(model.parameters()).dtype
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 1

    print("--- KV bytes, from the inside out ------------------------------------")
    per_token_per_layer = 2 * num_kv_heads * head_dim  # the 2 is K and V
    per_token = per_token_per_layer * num_layers
    print(f"  dtype as loaded       {dtype}  ({bytes_per_elem} bytes/element)")
    print(f"  one token, one layer  2 (K,V) x {num_kv_heads} heads x {head_dim} dim"
          f" = {per_token_per_layer:,} elements")
    print(f"  one token, all layers {per_token_per_layer:,} x {num_layers} layers"
          f" = {per_token:,} elements = {per_token * bytes_per_elem / 1024:.1f} KiB")
    print(f"  one BLOCK ({BLOCK_SIZE} tokens) = "
          f"{per_token * BLOCK_SIZE * bytes_per_elem / 1024:.1f} KiB")
    print()

    print("--- What that buys you at scale --------------------------------------")
    for budget_gb in (1, 2, 4):
        budget = budget_gb * 1024**3
        tokens = budget // (per_token * bytes_per_elem)
        blocks = tokens // BLOCK_SIZE
        print(f"  {budget_gb} GiB of KV cache = {tokens:,} tokens = {blocks:,} blocks"
              f"  (~{tokens // 1024:,} sequences of 1024 tokens)")
    print()
    print("  Now redo that mentally for a 7B model: ~28 layers, 4 KV heads,")
    print("  head_dim 128 -> roughly 15x this per token. That is why M4's")
    print("  quantization and M3's paging are the same conversation.")


if __name__ == "__main__":
    import torch  # noqa: E402  (imported late so the header prints fast)

    main()
