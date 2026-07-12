import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.sandbox import guided_sandbox, show_example

st.title("🎚️ Activation Functions")
st.markdown(
    """
Before stacking neurons into layers, one question decides everything: what
happens between the layers? The **activation function** is the little
nonlinear squash each neuron applies to its score — the σ in last page's
neuron. It looks like a detail. It is the entire reason deep learning works.
"""
)

st.header("1 · Why bother? Because linear + linear = still linear")
st.markdown(
    """
Suppose we stack two *linear* layers — layer 1 computes a new pair of
features from the inputs, layer 2 combines those. Feels like it should be
more powerful than one layer… but multiply it out and the stack **collapses
into a single linear layer**. Below, live: a random 2-layer linear network,
the single layer it algebraically collapses to, and proof they agree on
every input:
"""
)
show_example(
    '''import numpy as np
rng = np.random.default_rng(0)

W1 = rng.normal(size=(2, 2));  b1 = rng.normal(size=2)   # layer 1
W2 = rng.normal(size=(1, 2));  b2 = rng.normal(size=1)   # layer 2

def two_linear_layers(x):
    hidden = W1 @ x + b1          # first "layer" (no squash!)
    return W2 @ hidden + b2       # second "layer"

# Multiply the algebra out: W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
W_collapsed = W2 @ W1
b_collapsed = W2 @ b1 + b2

for x in [np.array([1.0, 2.0]), np.array([-3.0, 0.5]),
          np.array([10.0, -7.0])]:
    deep = two_linear_layers(x)
    flat = W_collapsed @ x + b_collapsed
    print(f"x={x}:  2-layer {deep.round(6)}  1-layer {flat.round(6)}")''',
    """
- `W1 @ x + b1` — a whole layer of neurons in one line: `@` is matrix multiplication, so each row of `W1` is one neuron's weights. (The forward-pass page dwells on this properly.)
- `W_collapsed = W2 @ W1` — the two weight matrices multiply into one. The printed lines agree to every decimal on all three inputs: stacking bought **nothing**.
- The fix: put a *nonlinear* function between the layers. Then the algebra can't collapse, and each extra layer genuinely adds expressive power. That function is the activation.
""",
)

st.header("2 · Meet the squashes")
st.markdown(
    "Drag the input and switch functions. Left: the activation itself. "
    "Right: its **derivative** — how much a nudge to the input changes the "
    "output. Park the marker far from zero and watch the derivative die; "
    "this single observation explains half of deep-learning history."
)

act = st.radio("activation", ["sigmoid", "tanh", "ReLU", "Leaky ReLU"],
               horizontal=True)
x_in = st.slider("input z", -6.0, 6.0, 1.5, 0.1)

zs = np.linspace(-6, 6, 400)
FUNCS = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),
                lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "ReLU": (lambda z: np.maximum(0, z),
             lambda z: (z > 0).astype(float)),
    "Leaky ReLU": (lambda z: np.where(z > 0, z, 0.05 * z),
                   lambda z: np.where(z > 0, 1.0, 0.05)),
}
f, df = FUNCS[act]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(zs, f(zs), color="#1565c0", linewidth=2.2)
axes[0].scatter([x_in], [f(np.array([x_in]))[0] if act != "tanh"
                         else f(x_in)], s=110, color="#e65100", zorder=5)
axes[0].set_title(f"{act}(z) — output at z={x_in:.1f}: "
                  f"{float(f(np.array([x_in]))[0]):.3f}", fontsize=10)
axes[0].grid(alpha=0.25)
axes[0].set_xlabel("z")
axes[1].plot(zs, df(zs), color="#6a1b9a", linewidth=2.2)
axes[1].scatter([x_in], [float(df(np.array([x_in]))[0])], s=110,
                color="#e65100", zorder=5)
axes[1].set_title(f"derivative — at z={x_in:.1f}: "
                  f"{float(df(np.array([x_in]))[0]):.3f}", fontsize=10)
axes[1].grid(alpha=0.25)
axes[1].set_xlabel("z")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
The personalities, and why you'd pick each:

| | shape | derivative | the catch |
|---|---|---|---|
| **sigmoid** | squashes to (0, 1) | tiny everywhere (max 0.25), ~0 in the tails | **saturates**: a confident neuron stops learning. Today: output layers only (when you want a probability) |
| **tanh** | squashes to (−1, 1), zero-centred | up to 1 at z=0, but still dies in the tails | the improved sigmoid of the 1990s; still saturates |
| **ReLU** | `max(0, z)` — literally "negative → 0" | exactly 1 when positive, 0 when negative | the modern default: cheap, doesn't saturate for positive z. Catch: a neuron stuck negative gets derivative 0 forever ("dead ReLU") |
| **Leaky ReLU** | ReLU with a small slope (here 0.05) for z < 0 | never exactly 0 | the dead-neuron insurance policy |

Why the derivative panel matters so much: training (backpropagation, two
pages from now) works by passing blame *backwards through* each activation,
**multiplying by this derivative at every layer**. Chain ten sigmoid layers
and blame gets multiplied by ≤ 0.25 ten times — it arrives microscopically
small at the early layers, which then barely learn. That's the **vanishing
gradient problem**, and ReLU's flat derivative of 1 is the blunt fix that
made deep networks trainable. File this away: the DeepSurv paper uses ReLU
and its cousin **SELU** — a self-normalising relative from the same
anti-vanishing family — precisely for these reasons.
"""
)

st.header("3 · Softmax: the team squash for multi-way choices")
st.markdown(
    """
Sigmoid answers yes/no. When a network must choose among **several**
classes, its final layer produces one raw score (a *logit*) per class, and
**softmax** converts the whole score vector into a probability distribution:

$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

Exponentiate every score (making them positive and stretching gaps), then
divide by the total (making them sum to 1). Try it — you're setting the raw
scores of a machine-fault classifier deciding *which component* is about to
fail:
"""
)
c1, c2, c3 = st.columns(3)
z_bearing = c1.slider("score: bearing fault", -4.0, 4.0, 2.0, 0.1)
z_seal = c2.slider("score: seal fault", -4.0, 4.0, 0.5, 0.1)
z_motor = c3.slider("score: motor fault", -4.0, 4.0, -1.0, 0.1)

logits = np.array([z_bearing, z_seal, z_motor])
expz = np.exp(logits)
probs = expz / expz.sum()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
names = ["bearing", "seal", "motor"]
axes[0].bar(names, logits, color="#90a4ae")
axes[0].axhline(0, color="#546e7a", linewidth=0.8)
axes[0].set_title("raw scores (logits) — any real numbers", fontsize=10)
axes[0].grid(alpha=0.25, axis="y")
axes[1].bar(names, probs, color=["#1565c0", "#e65100", "#2e7d32"])
axes[1].set_ylim(0, 1)
axes[1].set_title(f"after softmax — probabilities, sum = "
                  f"{probs.sum():.3f}", fontsize=10)
for i, p in enumerate(probs):
    axes[1].text(i, p + 0.02, f"{p:.1%}", ha="center", fontsize=9)
axes[1].grid(alpha=0.25, axis="y")
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Play with two behaviours worth internalising:

- **Only gaps matter.** Push all three sliders up by the same amount — the
  probabilities don't move (the shared shift cancels in the ratio).
- **Exponentials exaggerate.** A score lead of 2 becomes a probability lead
  of ~7×, because $e^2 \\approx 7.4$. Softmax is a "soft" argmax: the leader
  takes most (not all) of the mass.

**Why this matters for your dissertation, in one sentence:** DeepHit's
output layer is a softmax over *time intervals* — for equipment, read "the
probability the unit fails in month 1, month 2, …" — one probability per
time bin, guaranteed to sum to a proper distribution. When you meet it in
Section 7, it will be exactly the bar chart above with time bins as the
categories.
"""
)

st.header("4 · In code")
show_example(
    '''import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max()               # subtract the max: overflow protection
    e = np.exp(z)
    return e / e.sum()

scores = np.array([2.0, 0.5, -1.0])
print("relu of [-2, -0.5, 0, 3]:", relu(np.array([-2, -0.5, 0, 3.0])))
print("softmax:", softmax(scores).round(4), "sum:",
      softmax(scores).sum())
print("shift-invariance: softmax(scores + 100) =",
      softmax(scores + 100).round(4))''',
    """
- `np.maximum(0, z)` — elementwise max against 0: the whole of ReLU. (Note `np.maximum`, the two-array version — not `np.max`, which collapses an array to one number.)
- `z - z.max()` — the standard safety trick: `np.exp(1000)` overflows to infinity, but softmax only cares about *gaps* between scores, so shifting everything down by the max changes nothing mathematically and keeps every exponent ≤ 0. All real implementations do this.
- The last line proves the shift-invariance claim from the sliders: +100 to every score, identical probabilities out.
""",
)

guided_sandbox(
    key="n2",
    steps="""
1. **Step 1** — implement `sigmoid(z)`, `tanh_act(z)` (use `np.tanh`),
   `relu(z)`, and `softmax(z)` (with the subtract-max trick). Print each
   applied to the test vector `z_test`.
2. **Step 2** — verify softmax honestly: print its sum (≈ 1.0) and check
   `softmax(z_test)` equals `softmax(z_test + 50)` (use `np.allclose`).
3. **Step 3** — derivatives by nudge test (no calculus needed): for
   `z0 = 3.0`, print `(sigmoid(z0 + 1e-6) - sigmoid(z0 - 1e-6)) / 2e-6`
   and the same for z0 = 0. Compare with the derivative panel's story:
   big at 0, tiny at 3.
4. **Step 4 (stretch)** — chain effect: print `0.25 ** n` for n = 2, 5, 10
   layers. That's the best-case blame signal surviving n sigmoid layers —
   the vanishing gradient in one line of arithmetic.
""",
    setup_code='''import numpy as np

z_test = np.array([2.0, 0.5, -1.0, -3.0])
print("test vector:", z_test)

# Step 1: sigmoid, tanh_act, relu, softmax (subtract-max trick!)


# Step 2: softmax sums to 1, and is shift-invariant (np.allclose)


# Step 3: nudge-test the sigmoid derivative at z0 = 3.0 and z0 = 0.0


# Step 4 (stretch): 0.25 ** n for n in [2, 5, 10]
''',
    solution_code='''import numpy as np

z_test = np.array([2.0, 0.5, -1.0, -3.0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh_act(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

print("sigmoid:", sigmoid(z_test).round(4))
print("tanh:   ", tanh_act(z_test).round(4))
print("relu:   ", relu(z_test))
print("softmax:", softmax(z_test).round(4))

print("softmax sum:", softmax(z_test).sum())
print("shift-invariant:", np.allclose(softmax(z_test),
                                      softmax(z_test + 50)))

for z0 in [3.0, 0.0]:
    slope = (sigmoid(z0 + 1e-6) - sigmoid(z0 - 1e-6)) / 2e-6
    print(f"sigmoid slope at z={z0}: {slope:.4f}")

for n in [2, 5, 10]:
    print(f"blame surviving {n:2d} sigmoid layers (best case): "
          f"{0.25 ** n:.10f}")''',
)
