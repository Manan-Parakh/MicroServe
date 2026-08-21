import torch, math
import torch.nn.functional as F
torch.manual_seed(0)

d_model, n_q, n_kv, hd = 64, 8, 2, 8   # 8 query heads, 2 kv heads -> GQA group 4
S = 5

x  = torch.randn(1, S, d_model)
Wq = torch.randn(d_model, n_q * hd) / math.sqrt(d_model)
Wk = torch.randn(d_model, n_kv * hd) / math.sqrt(d_model)
Wv = torch.randn(d_model, n_kv * hd) / math.sqrt(d_model)
Wo = torch.randn(n_q * hd, d_model) / math.sqrt(n_q * hd)

def attention(x, k_cache=None, v_cache=None):
    B, T, _ = x.shape
    q = (x @ Wq).view(B, T, n_q,  hd).transpose(1, 2)   # [B, n_q,  T, hd]
    k = (x @ Wk).view(B, T, n_kv, hd).transpose(1, 2)   # [B, n_kv, T, hd]
    v = (x @ Wv).view(B, T, n_kv, hd).transpose(1, 2)

    if k_cache is not None:
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
    L = k.shape[2]

    kx = k.repeat_interleave(n_q // n_kv, dim=1)        # [B, n_q, L, hd]
    vx = v.repeat_interleave(n_q // n_kv, dim=1)

    scores = (q @ kx.transpose(-2, -1)) / math.sqrt(hd) # [B, n_q, T, L]
    mask = torch.ones(T, L, dtype=torch.bool).tril(diagonal=L - T)
    scores = scores.masked_fill(~mask, float("-inf"))
    w = scores.softmax(-1)
    out = (w @ vx).transpose(1, 2).reshape(B, T, n_q * hd) @ Wo
    return out, k, v, scores

# --- prefill: all S tokens at once ---
out_pre, k_pre, v_pre, scores = attention(x)
print("PREFILL")
print("  x      ", tuple(x.shape))
print("  scores ", tuple(scores.shape))
print("  out    ", tuple(out_pre.shape))

# --- check against PyTorch's own kernel ---
q = (x @ Wq).view(1, S, n_q, hd).transpose(1, 2)
ref = F.scaled_dot_product_attention(q, k_pre, v_pre, is_causal=True, enable_gqa=True)
ref = ref.transpose(1, 2).reshape(1, S, n_q * hd) @ Wo
print("  max diff vs SDPA:", (out_pre - ref).abs().max().item())

# --- decode: token S-1 alone, with cache of the first S-1 ---
out_p2, k_p2, v_p2, _ = attention(x[:, :S-1])
out_dec, k_d, v_d, sc_d = attention(x[:, S-1:], k_p2, v_p2)
print("DECODE")
print("  x      ", tuple(x[:, S-1:].shape))
print("  scores ", tuple(sc_d.shape))
print("  out    ", tuple(out_dec.shape))
print("  max diff vs prefill's last row:",
      (out_dec[:, 0] - out_pre[:, -1]).abs().max().item())