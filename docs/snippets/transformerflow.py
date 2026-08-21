"""
One decoder layer, every number printed. d_model=4, 2 heads, head_dim=2.
Small enough that you can check the arithmetic by hand.
"""
import torch, math
torch.set_printoptions(precision=3, sci_mode=False)

VOCAB = ["the", "cat", "sat", "on", "mat", "."]
d_model, n_heads, head_dim, d_ff = 4, 2, 2, 8

# ---------- weights (hand-chosen, not trained) ----------
E = torch.tensor([                     # token embeddings [vocab, d_model]
    [ 1.0,  0.0,  1.0,  0.0],   # the
    [ 0.0,  1.0,  1.0,  0.0],   # cat
    [ 1.0,  1.0,  0.0,  1.0],   # sat
    [ 0.0,  0.0,  1.0,  1.0],   # on
    [ 1.0,  0.0,  0.0,  1.0],   # mat
    [ 0.0,  1.0,  0.0,  1.0],   # .
])
Wq = torch.tensor([[ 1.,0., 0.,1.],[ 0.,1., 1.,0.],[ 1.,0., 0.,0.],[ 0.,0., 1.,1.]]) * 0.5
Wk = torch.tensor([[ 0.,1., 1.,0.],[ 1.,0., 0.,1.],[ 0.,1., 1.,0.],[ 1.,0., 0.,1.]]) * 0.5
Wv = torch.tensor([[ 1.,0., 0.,0.],[ 0.,1., 0.,0.],[ 0.,0., 1.,0.],[ 0.,0., 0.,1.]]) * 1.0
Wo = torch.tensor([[ 1.,0., 1.,0.],[ 0.,1., 0.,1.],[ 1.,0., 0.,1.],[ 0.,1., 1.,0.]]) * 0.5
Wg = torch.randn(d_model, d_ff, generator=torch.Generator().manual_seed(1)) * 0.5
Wu = torch.randn(d_model, d_ff, generator=torch.Generator().manual_seed(2)) * 0.5
Wd = torch.randn(d_ff, d_model, generator=torch.Generator().manual_seed(3)) * 0.5

def rmsnorm(x, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def rope(v, pos):
    """head_dim=2 -> exactly one 2D rotation by angle = pos * 1.0 radians."""
    a = torch.as_tensor(float(pos))
    c, s = torch.cos(a), torch.sin(a)
    x0, x1 = v[..., 0], v[..., 1]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)

def banner(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)

# ---------- 1. tokens -> residual stream ----------
ids = [0, 1, 2]                        # "the cat sat"
banner("1. EMBEDDING")
x = E[ids]
print("tokens:", [VOCAB[i] for i in ids], " ids:", ids)
print("x  [S=3, d=4]\n", x)

# ---------- 2. pre-attention norm ----------
banner("2. RMSNORM (before attention)")
print("rms per token:", x.pow(2).mean(-1).sqrt())
xn = rmsnorm(x)
print("xn\n", xn)

# ---------- 3. Q, K, V projections ----------
banner("3. Q K V PROJECTIONS")
q, k, v = xn @ Wq, xn @ Wk, xn @ Wv
print("q = xn @ Wq\n", q)
print("k = xn @ Wk\n", k)
print("v = xn @ Wv\n", v)

# ---------- 4. split into heads ----------
banner("4. SPLIT INTO 2 HEADS OF head_dim=2")
qh = q.view(3, n_heads, head_dim).transpose(0, 1)   # [head, S, hd]
kh = k.view(3, n_heads, head_dim).transpose(0, 1)
vh = v.view(3, n_heads, head_dim).transpose(0, 1)
for h in range(n_heads):
    print(f"head {h}: q\n{qh[h]}\n         k\n{kh[h]}")

# ---------- 5. RoPE ----------
banner("5. ROPE (rotate q,k by position angle)")
qr = torch.stack([torch.stack([rope(qh[h, p], p) for p in range(3)]) for h in range(n_heads)])
kr = torch.stack([torch.stack([rope(kh[h, p], p) for p in range(3)]) for h in range(n_heads)])
print("angles (radians):", [0, 1, 2])
print("head 0 q after RoPE\n", qr[0])
print("head 0 k after RoPE\n", kr[0])
print("note: V is NOT rotated")

# ---------- 6. attention scores ----------
banner("6. SCORES, MASK, SOFTMAX")
mask = torch.ones(3, 3).tril().bool()
outs = []
for h in range(n_heads):
    s = (qr[h] @ kr[h].T) / math.sqrt(head_dim)
    print(f"head {h}: raw scores  q_i . k_j / sqrt(2)\n{s}")
    sm = s.masked_fill(~mask, float("-inf"))
    print(f"head {h}: masked\n{sm}")
    w = sm.softmax(-1)
    print(f"head {h}: weights (rows sum to 1)\n{w}")
    o = w @ vh[h]
    print(f"head {h}: out = weights @ V\n{o}")
    outs.append(o)

# ---------- 7. merge heads + output projection ----------
banner("7. CONCAT HEADS -> Wo -> RESIDUAL")
concat = torch.cat(outs, dim=-1)
print("concat\n", concat)
attn_out = concat @ Wo
print("attn_out = concat @ Wo\n", attn_out)
x1 = x + attn_out
print("x1 = x + attn_out   <-- residual\n", x1)

# ---------- 8. FFN ----------
banner("8. FFN (SwiGLU) + RESIDUAL")
x1n = rmsnorm(x1)
gate, up = x1n @ Wg, x1n @ Wu
h = torch.nn.functional.silu(gate) * up
ffn_out = h @ Wd
print("gate = x1n @ Wg  [3, 8]\n", gate)
print("h = silu(gate) * up  [3, 8]\n", h)
print("ffn_out = h @ Wd\n", ffn_out)
x2 = x1 + ffn_out
print("x2 = x1 + ffn_out   <-- layer output, same shape as input\n", x2)

# ---------- 9. logits ----------
banner("9. FINAL NORM -> LM HEAD -> PREDICTION")
logits = rmsnorm(x2) @ E.T          # weight tying: lm_head = embedding^T
print("logits [3, 6]\n", logits)
probs = logits.softmax(-1)
print("probs for LAST position\n", probs[-1])
nxt = int(probs[-1].argmax())
print("predicted next token:", VOCAB[nxt])
print("(rows 0 and 1 are also predictions -- for positions 1 and 2 -- and are discarded)")

# ---------- 10. decode step ----------
banner("10. DECODE: feed 'on' alone, with cache")
new = E[[3]]                                   # "on", position 3
newn = rmsnorm(new)
qn, kn, vn = newn @ Wq, newn @ Wk, newn @ Wv
qnh = qn.view(1, n_heads, head_dim).transpose(0, 1)
knh = kn.view(1, n_heads, head_dim).transpose(0, 1)
vnh = vn.view(1, n_heads, head_dim).transpose(0, 1)
qnr = torch.stack([rope(qnh[h, 0], 3) for h in range(n_heads)]).unsqueeze(1)
knr = torch.stack([rope(knh[h, 0], 3) for h in range(n_heads)]).unsqueeze(1)
k_all = torch.cat([kr, knr], dim=1)            # cache grew 3 -> 4
v_all = torch.cat([vh, vnh], dim=1)
print("K cache was [2,3,2], now", tuple(k_all.shape))
outs2 = []
for hh in range(n_heads):
    s = (qnr[hh] @ k_all[hh].T) / math.sqrt(head_dim)
    print(f"head {hh}: scores [1,4] (no mask needed)\n{s}")
    w = s.softmax(-1)
    print(f"head {hh}: weights\n{w}")
    outs2.append(w @ v_all[hh])
a2 = torch.cat(outs2, dim=-1) @ Wo
x1d = new + a2
x1dn = rmsnorm(x1d)
x2d = x1d + (torch.nn.functional.silu(x1dn @ Wg) * (x1dn @ Wu)) @ Wd
print("decode logits\n", rmsnorm(x2d) @ E.T)
print("\nsame weights, same ops, one row instead of three.")