import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch

from utils.mockdata import pumps_xor
from utils.sandbox import guided_sandbox, show_example

st.title("➡️ The Forward Pass")
st.markdown(
    """
One neuron could only draw a straight line, and the XOR pump batch defeated
it. The cure is to **stack neurons into layers**: a *hidden layer* invents
new features from the raw sensors, and the output neuron classifies using
those invented features instead of the originals. Running data through the
stack — inputs → hidden → output — is the **forward pass**.

Our network for this page (the classic minimal XOR solver):

- **2 inputs**: vibration, temperature
- **1 hidden layer of 2 neurons**, each with a tanh squash
- **1 output neuron** with a sigmoid squash → P(fail)

That's 9 numbers in total: 4 hidden weights + 2 hidden biases + 2 output
weights + 1 output bias. The weights below were **trained for real** (by the
training loop you'll write two pages from now) — this network genuinely
solves XOR, and we're going to watch it think.
"""
)


@st.cache_resource(show_spinner="Training the tiny XOR network…")
def train_xor_net():
    """Train the 2-2-1 net with plain gradient descent. Returns the weights
    and the final accuracy — everything on this page comes from this fit."""
    X, y = pumps_xor()
    Xs = (X - 5.0) / 2.5                       # centre and scale the sensors
    rng = np.random.default_rng(11)
    W1 = rng.normal(0, 1.5, (2, 2))
    b1 = rng.normal(0, 1.5, 2)
    W2 = rng.normal(0, 1.5, 2)
    b2 = 0.0
    lr = 0.5
    for _ in range(8000):
        Z1 = Xs @ W1 + b1
        H = np.tanh(Z1)
        Z2 = H @ W2 + b2
        P = 1 / (1 + np.exp(-Z2))
        dZ2 = (P - y) / len(y)
        gW2 = H.T @ dZ2
        gb2 = dZ2.sum()
        dH = np.outer(dZ2, W2) * (1 - H ** 2)
        gW1 = Xs.T @ dH
        gb1 = dH.sum(axis=0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    acc = np.mean(((1 / (1 + np.exp(-(np.tanh(Xs @ W1 + b1) @ W2 + b2))))
                   >= 0.5).astype(int) == y)
    return W1, b1, W2, b2, acc


W1, b1, W2, b2, net_acc = train_xor_net()
X, y = pumps_xor()

st.header("1 · Watch the numbers flow")
st.markdown(
    "Set a pump's two sensor readings. Every number in the diagram is "
    "computed live by the real network; **edge thickness = the magnitude of "
    "the signal travelling along it**, and red/blue = negative/positive."
)
c1, c2 = st.columns(2)
vib = c1.slider("vibration", 0.0, 10.0, 8.0, 0.1)
temp = c2.slider("temperature", 0.0, 10.0, 2.0, 0.1)

x = np.array([(vib - 5.0) / 2.5, (temp - 5.0) / 2.5])   # same scaling
z1 = x @ W1 + b1
h = np.tanh(z1)
z2 = h @ W2 + b2
p = 1 / (1 + np.exp(-z2))

fig, ax = plt.subplots(figsize=(10, 4.6))
pos_in = [(1.0, 3.1), (1.0, 1.3)]
pos_hid = [(5.0, 3.4), (5.0, 1.0)]
pos_out = (9.0, 2.2)


def draw_edge(p0, p1, value, label_offset=0.0):
    colour = "#1565c0" if value >= 0 else "#c62828"
    lw = 0.6 + 3.4 * min(abs(value), 3.0) / 3.0
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 color=colour, linewidth=lw, alpha=0.85))
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + label_offset
    ax.text(mx, my, f"{value:+.2f}", fontsize=8, color=colour, ha="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))


# input -> hidden: the signal travelling each wire is x_i * W1[i, j]
for i, pin in enumerate(pos_in):
    for j, phid in enumerate(pos_hid):
        start = (pin[0] + 0.45, pin[1])
        end = (phid[0] - 0.5, phid[1])
        draw_edge(start, end, x[i] * W1[i, j], 0.12)
# hidden -> output: signal is h_j * W2[j]
for j, phid in enumerate(pos_hid):
    draw_edge((phid[0] + 0.5, phid[1]), (pos_out[0] - 0.55, pos_out[1]),
              h[j] * W2[j], 0.12)

for i, (px, py) in enumerate(pos_in):
    ax.add_patch(Circle((px, py), 0.45, facecolor="#e3f2fd",
                        edgecolor="#1565c0", linewidth=2))
    ax.text(px, py, f"{x[i]:+.2f}", ha="center", va="center", fontsize=10,
            fontweight="bold")
    ax.text(px, py - 0.68, ["vibration", "temperature"][i], ha="center",
            fontsize=8, color="#546e7a")
    ax.text(px, py + 0.62, f"raw {[vib, temp][i]:.1f} → scaled", ha="center",
            fontsize=7, color="#90a4ae")
for j, (px, py) in enumerate(pos_hid):
    ax.add_patch(Circle((px, py), 0.5, facecolor="#fff3e0",
                        edgecolor="#e65100", linewidth=2))
    ax.text(px, py + 0.11, f"z={z1[j]:+.2f}", ha="center", fontsize=8)
    ax.text(px, py - 0.16, f"tanh→{h[j]:+.2f}", ha="center", fontsize=9,
            fontweight="bold")
    ax.text(px, py - 0.72, f"hidden neuron {j + 1}\n(bias {b1[j]:+.2f})",
            ha="center", fontsize=7.5, color="#546e7a")
ax.add_patch(Circle(pos_out, 0.55,
                    facecolor="#ffcdd2" if p >= 0.5 else "#c8e6c9",
                    edgecolor="#37474f", linewidth=2))
ax.text(pos_out[0], pos_out[1] + 0.12, f"z={z2:+.2f}", ha="center",
        fontsize=8)
ax.text(pos_out[0], pos_out[1] - 0.17, f"σ→{p:.2f}", ha="center", fontsize=10,
        fontweight="bold")
ax.text(pos_out[0], pos_out[1] - 0.62, f"P(fail)\n(bias {b2:+.2f})",
        ha="center", va="top", fontsize=7.5, color="#546e7a")
ax.set_xlim(0, 10.2)
ax.set_ylim(0.1, 4.3)
ax.axis("off")
ax.set_title(f"forward pass — verdict: {'FAIL' if p >= 0.5 else 'PASS'} "
             f"(P = {p:.3f})", fontsize=11)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    f"""
Follow the arithmetic yourself — it's only the last page's neuron, three
times over:

- **hidden neuron 1**: z = ({x[0]:+.2f})({W1[0, 0]:+.2f}) +
  ({x[1]:+.2f})({W1[1, 0]:+.2f}) + ({b1[0]:+.2f}) = **{z1[0]:+.2f}**,
  then tanh → **{h[0]:+.2f}**
- **hidden neuron 2**: z = ({x[0]:+.2f})({W1[0, 1]:+.2f}) +
  ({x[1]:+.2f})({W1[1, 1]:+.2f}) + ({b1[1]:+.2f}) = **{z1[1]:+.2f}**,
  then tanh → **{h[1]:+.2f}**
- **output**: z = ({h[0]:+.2f})({W2[0]:+.2f}) + ({h[1]:+.2f})({W2[1]:+.2f})
  + ({b2:+.2f}) = **{z2:+.2f}**, then σ → **{p:.3f}**

Try the four corners of the XOR pattern — (8, 2), (2, 8), (8, 8), (2, 2) —
and watch the hidden neurons' outputs `h` rearrange themselves so that the
output neuron can separate the cases with a simple weighted sum. That
rearrangement *is* the hidden layer earning its keep.
"""
)

st.header("2 · What the hidden layer actually invented")
st.markdown(
    "Here's the payoff, and the single most illuminating picture in this "
    "section. **Left**: the raw sensor space — the XOR pattern that no line "
    "can split. **Right**: the same pumps plotted in the *hidden layer's* "
    "coordinates (h₁, h₂) — the network's own invented feature space. Both "
    "computed live from the trained weights."
)

Xs_all = (X - 5.0) / 2.5
H_all = np.tanh(Xs_all @ W1 + b1)

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for cls, colour, lab in [(0, "#2e7d32", "passed"), (1, "#c62828", "failed")]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=colour, s=22, edgecolor="white",
                   label=lab)
    ax.scatter([vib], [temp], marker="*", s=340, c="#ffd600",
               edgecolor="black", zorder=6, label="your pump")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("vibration")
    ax.set_ylabel("temperature")
    ax.set_title("raw sensors: no line works", fontsize=10)
    ax.legend(fontsize=7)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for cls, colour in [(0, "#2e7d32"), (1, "#c62828")]:
        m = y == cls
        ax.scatter(H_all[m, 0], H_all[m, 1], c=colour, s=22,
                   edgecolor="white")
    # the output neuron's boundary IS a line in this space: h·W2 + b2 = 0
    hs = np.linspace(-1.05, 1.05, 10)
    if abs(W2[1]) > 1e-6:
        ax.plot(hs, -(W2[0] * hs + b2) / W2[1], color="#37474f",
                linewidth=2.2, label="output neuron's line")
    ax.scatter([h[0]], [h[1]], marker="*", s=340, c="#ffd600",
               edgecolor="black", zorder=6)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("h₁ (hidden neuron 1)")
    ax.set_ylabel("h₂ (hidden neuron 2)")
    ax.set_title("the hidden layer's invented space:\na line works here!",
                 fontsize=10)
    ax.legend(fontsize=7)
    st.pyplot(fig)
    plt.close(fig)

st.success(
    f"**This is what deep learning *is*.** The hidden layer bent the space "
    f"until the problem became linearly separable, then the output neuron — "
    f"still just the humble line-drawer from page 1 — solved it trivially. "
    f"Accuracy on the XOR batch: **{net_acc:.0%}** (versus ~50% for the "
    "single neuron). Compare with the SVM page: there *we* hand-crafted the "
    "extra dimension, and the kernel trick supplied a fixed recipe. Here "
    "the network **learns its own transformation from the data** — which is "
    "why the same architecture works for sensors, images, and (in Section 6) "
    "risk scores."
)

st.header("3 · The forward pass in code — matrices, not loops")
show_example(
    '''import numpy as np

# A layer's worth of weights: one COLUMN per neuron.
W1 = np.array([[ 1.5, -1.2],      # how vibration feeds hidden 1, hidden 2
               [-1.4,  1.3]])     # how temperature feeds hidden 1, hidden 2
b1 = np.array([0.2, 0.1])
W2 = np.array([2.0, 2.0])
b2 = -1.0

def forward(x):
    z1 = x @ W1 + b1              # both hidden scores in ONE matrix multiply
    h = np.tanh(z1)               # squash, elementwise
    z2 = h @ W2 + b2              # the output score
    p = 1 / (1 + np.exp(-z2))     # squash into a probability
    return z1, h, z2, p

x = np.array([1.0, -1.0])         # a scaled pump: high vib, low temp
z1, h, z2, p = forward(x)
print("hidden scores z1:", z1.round(3))
print("hidden outputs h:", h.round(3))
print("output score z2: ", round(z2, 3), " P(fail):", round(p, 3))

# A BATCH of pumps: identical code, X is (n_pumps, 2) instead of (2,)
X = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
H = np.tanh(X @ W1 + b1)          # (4, 2) - all four pumps at once
P = 1 / (1 + np.exp(-(H @ W2 + b2)))
print("batch of 4 P(fail):", P.round(3))''',
    """
- `W1` is a **2×2 matrix**: entry `W1[i, j]` is the weight from input *i* into hidden neuron *j*. One column per neuron — that convention is worth burning in, because every framework uses it.
- `x @ W1 + b1` — the `@` (matrix multiply) computes *every* hidden neuron's score in one operation. This is the vectorization lesson from the NumPy page cashing in: no Python loop over neurons, ever. It's also why GPUs (which are matrix-multiply machines) transformed this field.
- `np.tanh(z1)` — applied elementwise to the whole vector; `b1` broadcasts.
- The batch block is the punchline: **the code does not change**. Feed a `(4, 2)` matrix of four pumps instead of one `(2,)` vector, and broadcasting handles it — `H` comes out `(4, 2)`, `P` comes out with 4 probabilities. Real training pushes hundreds of examples through this way at once.
""",
)

guided_sandbox(
    key="n3",
    steps="""
1. **Step 1** — write `forward(x)` returning `(z1, h, z2, p)` for the
   trained weights the setup code loads: `z1 = x @ W1 + b1`, `h = np.tanh(z1)`,
   `z2 = h @ W2 + b2`, `p = 1/(1+np.exp(-z2))`.
2. **Step 2** — run it on the four XOR corners (already scaled for you in
   `corners`) and print each one's `h` and `p`. Confirm the two "exactly one
   sensor high" corners come out as FAIL (p > 0.5) and the other two as PASS.
3. **Step 3** — do the whole batch in one go: `H = np.tanh(Xs @ W1 + b1)`,
   then the output layer, and print the network's accuracy against `y`.
4. **Step 4 (stretch)** — knock out hidden neuron 2 by zeroing its outgoing
   weight (`W2[1] = 0`) and re-measure accuracy. How much of the network's
   ability lived in that one neuron? (Restore it afterwards.)
""",
    setup_code='''import numpy as np
from utils.mockdata import pumps_xor

# The trained weights from this page's network (real values, rounded to 2dp):
W1 = np.array([[-1.33,  3.12],
               [ 1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90

X, y = pumps_xor()
Xs = (X - 5.0) / 2.5                     # the same scaling the net was trained on
corners = np.array([[1.2, -1.2],         # high vib, low temp  -> FAIL
                    [-1.2, 1.2],         # low vib, high temp  -> FAIL
                    [1.2, 1.2],          # both high           -> PASS
                    [-1.2, -1.2]])       # both low            -> PASS
print("weights loaded; batch shape:", Xs.shape)

# Step 1: define forward(x)


# Step 2: run the four corners, print h and p for each


# Step 3: the whole batch at once; print accuracy


# Step 4 (stretch): zero W2[1] and re-measure
''',
    solution_code='''import numpy as np
from utils.mockdata import pumps_xor

W1 = np.array([[-1.33,  3.12],
               [ 1.28, -2.89]])
b1 = np.array([-1.35, -2.83])
W2 = np.array([7.54, 6.55])
b2 = 4.90

X, y = pumps_xor()
Xs = (X - 5.0) / 2.5
corners = np.array([[1.2, -1.2], [-1.2, 1.2], [1.2, 1.2], [-1.2, -1.2]])

def forward(x):
    z1 = x @ W1 + b1
    h = np.tanh(z1)
    z2 = h @ W2 + b2
    p = 1 / (1 + np.exp(-z2))
    return z1, h, z2, p

names = ["high vib, low temp", "low vib, high temp", "both high", "both low"]
for corner, name in zip(corners, names):
    _, h, _, p = forward(corner)
    print(f"{name:20}: h={h.round(2)}  P(fail)={p:.3f}  "
          f"-> {'FAIL' if p >= 0.5 else 'PASS'}")

H = np.tanh(Xs @ W1 + b1)
P = 1 / (1 + np.exp(-(H @ W2 + b2)))
acc = np.mean((P >= 0.5).astype(int) == y)
print(f"\\nfull batch accuracy: {acc:.0%}")

W2_broken = W2.copy()
W2_broken[1] = 0.0
P2 = 1 / (1 + np.exp(-(H @ W2_broken + b2)))
print(f"with hidden neuron 2 silenced: "
      f"{np.mean((P2 >= 0.5).astype(int) == y):.0%} "
      "- it collapses back to a single-neuron model")''',
)
