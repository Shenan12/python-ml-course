import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.patches import Circle, FancyArrowPatch

from utils.sandbox import guided_sandbox, show_example

st.title("⬅️ Backpropagation")
st.markdown(
    """
The forward pass turned sensors into a prediction. Now the hard question:
the prediction was wrong by some amount — **which of the 9 weights should
change, and in which direction?**

You already know the shape of the answer. Gradient descent (Section 2) says:
compute the gradient of the loss with respect to each parameter, then step
downhill. The only new problem is *how to compute those gradients* when a
weight's influence travels through several layers before reaching the loss.

The answer is the **chain rule**, applied backwards through the network —
and that's all backpropagation is. Not a new algorithm; a bookkeeping
strategy. Two claims to hold on to:

1. **Blame flows backwards.** Start with "how wrong was the output?", then
   ask each layer "how much of that was *your* doing?", passing responsibility
   back through the weights, layer by layer.
2. **Each layer multiplies the incoming blame by two local things**: the
   derivative of its activation, and its weights. (That "multiply at every
   layer" is exactly why saturating activations cause vanishing gradients —
   the point flagged on the activations page.)
"""
)

st.header("1 · Blame flowing backwards, on one example")
st.markdown(
    "One pump, one forward pass, one backward pass — every number below is "
    "computed live. Set the pump and its true label, and watch the two "
    "sweeps: **blue arrows = numbers going forward**, **red arrows = blame "
    "(gradients) coming back**."
)

# a fixed small network (the XOR-style solver from the previous page)
W1 = np.array([[-1.33, 3.12], [1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90

c1, c2, c3 = st.columns(3)
vib = c1.slider("vibration", 0.0, 10.0, 8.0, 0.1)
temp = c2.slider("temperature", 0.0, 10.0, 2.0, 0.1)
truth = c3.radio("what actually happened", ["failed (y=1)", "passed (y=0)"],
                 index=0)
y_true = 1.0 if truth.startswith("failed") else 0.0

x = np.array([(vib - 5.0) / 2.5, (temp - 5.0) / 2.5])

# ---------- FORWARD ----------
z1 = x @ W1 + b1
h = np.tanh(z1)
z2 = h @ W2 + b2
p = 1 / (1 + np.exp(-z2))
loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

# ---------- BACKWARD (the chain rule, by hand) ----------
dz2 = p - y_true                     # dLoss/dz2  (the famous clean result)
gW2 = dz2 * h                        # dLoss/dW2
gb2 = dz2                            # dLoss/db2
dh = dz2 * W2                        # blame arriving at each hidden output
dz1 = dh * (1 - h ** 2)              # through tanh's derivative
gW1 = np.outer(x, dz1)               # dLoss/dW1  (outer product!)
gb1 = dz1                            # dLoss/db1

fig, ax = plt.subplots(figsize=(10, 5))
pos_in = [(1.0, 3.4), (1.0, 1.5)]
pos_hid = [(5.0, 3.7), (5.0, 1.2)]
pos_out = (9.0, 2.45)

for i, pin in enumerate(pos_in):
    for j, phid in enumerate(pos_hid):
        ax.add_patch(FancyArrowPatch((pin[0] + 0.45, pin[1]),
                                     (phid[0] - 0.5, phid[1]),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#90caf9", linewidth=1.4,
                                     alpha=0.9))
        # backward arrow, thickness = |gradient| for that weight
        g = gW1[i, j]
        ax.add_patch(FancyArrowPatch((phid[0] - 0.5, phid[1] - 0.22),
                                     (pin[0] + 0.45, pin[1] - 0.22),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#ef5350",
                                     linewidth=0.6 + 5 * min(abs(g), 0.5) / 0.5,
                                     alpha=0.9,
                                     connectionstyle="arc3,rad=0.1"))
        mx = (pin[0] + phid[0]) / 2
        my = (pin[1] + phid[1]) / 2
        ax.text(mx, my - 0.42, f"∂L/∂w={g:+.3f}", fontsize=7.5,
                color="#c62828", ha="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.8))
for j, phid in enumerate(pos_hid):
    ax.add_patch(FancyArrowPatch((phid[0] + 0.5, phid[1]),
                                 (pos_out[0] - 0.55, pos_out[1]),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#90caf9", linewidth=1.4))
    g = gW2[j]
    ax.add_patch(FancyArrowPatch((pos_out[0] - 0.55, pos_out[1] - 0.24),
                                 (phid[0] + 0.5, phid[1] - 0.24),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#ef5350",
                                 linewidth=0.6 + 5 * min(abs(g), 0.5) / 0.5,
                                 connectionstyle="arc3,rad=0.1"))
    ax.text((phid[0] + pos_out[0]) / 2, (phid[1] + pos_out[1]) / 2 - 0.45,
            f"∂L/∂w={g:+.3f}", fontsize=7.5, color="#c62828", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.8))

for i, (px, py) in enumerate(pos_in):
    ax.add_patch(Circle((px, py), 0.45, facecolor="#e3f2fd",
                        edgecolor="#1565c0", linewidth=2))
    ax.text(px, py, f"{x[i]:+.2f}", ha="center", va="center", fontsize=10,
            fontweight="bold")
    ax.text(px, py - 0.66, ["vibration", "temperature"][i], ha="center",
            fontsize=8, color="#546e7a")
for j, (px, py) in enumerate(pos_hid):
    ax.add_patch(Circle((px, py), 0.5, facecolor="#fff3e0",
                        edgecolor="#e65100", linewidth=2))
    ax.text(px, py + 0.1, f"h={h[j]:+.2f}", ha="center", fontsize=9,
            fontweight="bold")
    ax.text(px, py - 0.18, f"δ={dz1[j]:+.3f}", ha="center", fontsize=8,
            color="#c62828")
    ax.text(px, py - 0.74, f"hidden {j + 1}", ha="center", fontsize=7.5,
            color="#546e7a")
ax.add_patch(Circle(pos_out, 0.55, facecolor="#ffcdd2" if p >= 0.5
                    else "#c8e6c9", edgecolor="#37474f", linewidth=2))
ax.text(pos_out[0], pos_out[1] + 0.12, f"p={p:.3f}", ha="center", fontsize=9,
        fontweight="bold")
ax.text(pos_out[0], pos_out[1] - 0.18, f"δ={dz2:+.3f}", ha="center",
        fontsize=8, color="#c62828")
ax.text(pos_out[0], pos_out[1] - 0.8, f"truth y={y_true:.0f}", ha="center",
        fontsize=8, color="#546e7a")
ax.text(0.2, 4.55, "forward: inputs → prediction", fontsize=9,
        color="#1565c0")
ax.text(0.2, 4.25, "backward: blame → every weight", fontsize=9,
        color="#c62828")
ax.set_xlim(0, 10.3)
ax.set_ylim(0.2, 4.75)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("prediction p", f"{p:.3f}")
c2.metric("loss (log-loss)", f"{loss:.4f}")
st.markdown(
    f"""
Read the backward sweep from the right:

1. **At the output**, blame starts as `δ = p − y = {p:.3f} − {y_true:.0f} =
   {dz2:+.3f}`. This astonishingly clean formula (no sigmoid derivative in
   sight!) is what you get when a sigmoid output meets a log-loss — the two
   derivatives cancel. Read it plainly: **blame = how wrong you were, and in
   which direction.**
2. **The output weights' gradients** are `δ × h` — blame times the input
   that fed it. So a hidden neuron that shouted (large |h|) into a wrong
   answer takes proportionally more of the blame. Every gradient in a
   network has this shape: *local input × incoming blame*.
3. **Blame passes back through the weights** (`δ × W2`) and then **through
   tanh's derivative** (`× (1 − h²)`), giving each hidden neuron its own δ.
   Notice: if a hidden neuron is saturated (h near ±1), then 1 − h² ≈ 0 and
   **its blame is annihilated** — it learns nothing this step. That's the
   vanishing gradient, visible in a single number.
4. **The input weights' gradients** are `x × δ_hidden` — same shape again.

Flip the truth label and watch every red arrow reverse sign: the network is
being pushed the other way. Set the pump to a case it already gets right
with confidence, and watch all the blame shrink toward zero — **a confident,
correct network stops changing.** That's convergence.
"""
)

st.header("2 · Verifying the gradients three independent ways")
st.markdown(
    """
Hand-derived calculus is exactly where silent bugs live, so nobody sane
trusts it unchecked. We verify the same gradient three ways:

1. **The analytic formulas** above (what we coded by hand).
2. **A finite-difference nudge test** — no calculus at all: change a weight
   by a hair, see how much the loss moves, divide. (The same trick you used
   on the gradient-descent page.)
3. **PyTorch autograd** — the industrial machine, which builds a graph of
   every operation and differentiates it automatically.

All three, computed live on your current pump, for all 9 parameters:
"""
)


def loss_of(params, x, y_true):
    W1_, b1_, W2_, b2_ = params
    h_ = np.tanh(x @ W1_ + b1_)
    z2_ = h_ @ W2_ + b2_
    p_ = 1 / (1 + np.exp(-z2_))
    p_ = np.clip(p_, 1e-12, 1 - 1e-12)
    return -(y_true * np.log(p_) + (1 - y_true) * np.log(1 - p_))


# 2. finite differences
eps = 1e-6
fd = {}
for name, arr, grad in [("W1", W1, gW1), ("b1", b1, gb1),
                        ("W2", W2, gW2), ("b2", np.array([b2]),
                                          np.array([gb2]))]:
    flat = np.array(arr, dtype=float).ravel()
    out = np.zeros_like(flat)
    for i in range(flat.size):
        up, dn = flat.copy(), flat.copy()
        up[i] += eps
        dn[i] -= eps

        def rebuild(vals):
            if name == "W1":
                return (vals.reshape(2, 2), b1, W2, b2)
            if name == "b1":
                return (W1, vals, W2, b2)
            if name == "W2":
                return (W1, b1, vals, b2)
            return (W1, b1, W2, float(vals[0]))

        out[i] = ((loss_of(rebuild(up), x, y_true)
                   - loss_of(rebuild(dn), x, y_true)) / (2 * eps))
    fd[name] = out

# 3. PyTorch autograd
tW1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True, dtype=torch.float64)
tx = torch.tensor(x)
th = torch.tanh(tx @ tW1 + tb1)
tz2 = th @ tW2 + tb2
tloss = torch.nn.functional.binary_cross_entropy_with_logits(
    tz2, torch.tensor(y_true, dtype=torch.float64))
tloss.backward()

rows = []
labels = (["W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]"]
          + ["b1[0]", "b1[1]", "W2[0]", "W2[1]", "b2"])
analytic = np.concatenate([gW1.ravel(), gb1.ravel(), gW2.ravel(), [gb2]])
finite = np.concatenate([fd["W1"], fd["b1"], fd["W2"], fd["b2"]])
autograd = np.concatenate([tW1.grad.numpy().ravel(), tb1.grad.numpy().ravel(),
                           tW2.grad.numpy().ravel(),
                           np.array([tb2.grad.item()])])
for lab, a, f_, t in zip(labels, analytic, finite, autograd):
    rows.append({"parameter": lab, "1. our calculus": f"{a:+.6f}",
                 "2. nudge test": f"{f_:+.6f}",
                 "3. PyTorch autograd": f"{t:+.6f}",
                 "max disagreement": f"{max(abs(a - f_), abs(a - t)):.2e}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

max_err = float(np.max(np.abs(analytic - autograd)))
max_err_fd = float(np.max(np.abs(analytic - finite)))
if max_err < 1e-8 and max_err_fd < 1e-5:
    st.success(
        f"✅ **All three methods agree.** Our hand-derived gradients match "
        f"PyTorch's autograd to {max_err:.1e} and the calculus-free nudge "
        f"test to {max_err_fd:.1e} (the nudge test is only approximate by "
        "nature — it's limited by floating-point precision, so ~1e-8 "
        "agreement is as good as it gets). The backprop maths on this page "
        "is *verified*, not asserted."
    )
else:
    st.error(
        f"⚠️ The three methods disagree (analytic vs autograd: {max_err:.2e}, "
        f"analytic vs nudge: {max_err_fd:.2e}). Do not trust this page's "
        "gradient formulas until this is investigated."
    )

st.header("3 · The whole thing in code — 20 lines, no framework")
show_example(
    '''import numpy as np

W1 = np.array([[-1.33, 3.12], [1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90
x = np.array([1.2, -1.2])      # a scaled pump
y = 1.0                        # it failed

# ---- forward ----
z1 = x @ W1 + b1
h  = np.tanh(z1)
z2 = h @ W2 + b2
p  = 1 / (1 + np.exp(-z2))
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"prediction {p:.4f}, loss {loss:.4f}")

# ---- backward (the chain rule, layer by layer) ----
dz2 = p - y                    # sigmoid + log-loss cancel to this
gW2 = dz2 * h                  # local input x incoming blame
gb2 = dz2
dh  = dz2 * W2                 # blame passes back THROUGH the weights
dz1 = dh * (1 - h**2)          # ...and through tanh's derivative
gW1 = np.outer(x, dz1)         # local input x incoming blame, again
gb1 = dz1

print("gradient wrt W2:", gW2.round(4))
print("gradient wrt W1:\\n", gW1.round(4))

# ---- one gradient-descent step (Section 2, unchanged) ----
lr = 0.1
W1, b1, W2, b2 = W1 - lr*gW1, b1 - lr*gb1, W2 - lr*gW2, b2 - lr*gb2
h2 = np.tanh(x @ W1 + b1)
p2 = 1 / (1 + np.exp(-(h2 @ W2 + b2)))
print(f"after one step: prediction {p2:.4f} (was {p:.4f}) - moved toward 1")''',
    """
- The **forward** block is the previous page, unchanged.
- `dz2 = p - y` — the entire starting blame. It's a *number*, not a mystery: how wrong, and which way.
- `gW2 = dz2 * h` — every gradient in every neural network has this form: **the input that flowed in, times the blame that flowed back**. Memorise the shape and backprop stops being scary.
- `dh = dz2 * W2` — blame travels backwards along the same wires the signal came forwards on, scaled by the same weights. A weight that amplified a signal forwards amplifies the blame backwards.
- `dz1 = dh * (1 - h**2)` — `1 − tanh²` is tanh's derivative. **This is the multiplication that kills deep sigmoid/tanh networks**: it's ≤ 1, so blame shrinks at every layer it crosses.
- `np.outer(x, dz1)` — the outer product builds the whole 2×2 gradient matrix at once: element `[i, j]` = `x[i] * dz1[j]`, precisely "input i's value × neuron j's blame".
- The last block shows the payoff: gradients feed straight into the **same gradient-descent update** you wrote in Section 2. Backprop doesn't replace gradient descent — it *supplies* it with gradients.
""",
)

guided_sandbox(
    key="n4",
    steps="""
1. **Step 1** — write `forward(x)` returning `(h, p)` and a `loss(p, y)`
   using the log-loss `-(y*log(p) + (1-y)*log(1-p))`. Print both for the
   pump provided.
2. **Step 2** — write `backward(x, h, p, y)` returning `(gW1, gb1, gW2, gb2)`
   following the five lines from the snippet above (`dz2`, `gW2`, `gb2`,
   `dh`, `dz1`, `gW1`, `gb1`).
3. **Step 3** — **verify yourself with a nudge test**: pick one weight, say
   `W2[0]`. Compute `(loss(forward_with(W2[0]+1e-6)) -
   loss(forward_with(W2[0]-1e-6))) / 2e-6` and compare to your `gW2[0]`.
   They should agree to ~6 decimal places. Never trust a hand-derived
   gradient you haven't nudge-tested.
4. **Step 4 (stretch)** — the network is currently **confidently wrong**
   about this pump (it predicts ≈0.0003 when the truth is 1.0 — a loss of
   about 8.3). Take 50 gradient-descent steps on it (`lr=0.5`), printing the
   loss every 10, and watch the network talk itself out of its mistake. You
   have just trained a neural network with no framework at all.
""",
    setup_code='''import numpy as np

W1 = np.array([[-1.33, 3.12], [1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90

# A surprising unit: both sensors read high, yet it FAILED anyway. The
# network is currently confidently WRONG about it - perfect for watching
# gradient descent correct a mistake.
x = np.array([1.2, 1.2])
y = 1.0
print("start: W2 =", W2, " b2 =", b2)

# Step 1: forward(x) -> (h, p), and loss(p, y)


# Step 2: backward(x, h, p, y) -> (gW1, gb1, gW2, gb2)


# Step 3: nudge-test your gW2[0] against a finite difference


# Step 4 (stretch): 50 gradient descent steps, print loss every 10
''',
    solution_code='''import numpy as np

W1 = np.array([[-1.33, 3.12], [1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90
x = np.array([1.2, 1.2])
y = 1.0          # it failed, despite both sensors reading high

def forward(x, W1=W1, b1=b1, W2=W2, b2=b2):
    h = np.tanh(x @ W1 + b1)
    p = 1 / (1 + np.exp(-(h @ W2 + b2)))
    return h, p

def loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

h, p = forward(x)
print(f"prediction {p:.6f}, loss {loss(p, y):.6f}")

def backward(x, h, p, y, W2=W2):
    dz2 = p - y
    gW2 = dz2 * h
    gb2 = dz2
    dh = dz2 * W2
    dz1 = dh * (1 - h ** 2)
    gW1 = np.outer(x, dz1)
    gb1 = dz1
    return gW1, gb1, gW2, gb2

gW1, gb1, gW2, gb2 = backward(x, h, p, y)
print("gW2:", gW2.round(6))

eps = 1e-6
W2_up, W2_dn = W2.copy(), W2.copy()
W2_up[0] += eps
W2_dn[0] -= eps
fd = (loss(forward(x, W2=W2_up)[1], y)
      - loss(forward(x, W2=W2_dn)[1], y)) / (2 * eps)
print(f"nudge test for W2[0]: {fd:+.6f}   my gradient: {gW2[0]:+.6f}")
print("agree:", np.isclose(fd, gW2[0], atol=1e-5))

W1c, b1c, W2c, b2c = W1.copy(), b1.copy(), W2.copy(), b2
for step in range(50):
    h, p = forward(x, W1c, b1c, W2c, b2c)
    gW1, gb1, gW2, gb2 = backward(x, h, p, y, W2c)
    W1c -= 0.5 * gW1
    b1c -= 0.5 * gb1
    W2c -= 0.5 * gW2
    b2c -= 0.5 * gb2
    if step % 10 == 0:
        print(f"step {step:2d}: loss {loss(p, y):.4f}  p={p:.4f}")
print(f"final prediction: {forward(x, W1c, b1c, W2c, b2c)[1]:.4f} "
      "(target 1.0) - it corrected itself")''',
)
