import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from utils.sandbox import guided_sandbox, show_example

st.title("🗺️ The Landscape: CNNs, RNNs & Transformers")
st.markdown(
    """
Everything so far used a **fully-connected** network (`nn.Linear` layers):
every input touches every neuron. That's the right default when your inputs
are a flat list of features — vibration, temperature, pressure — with no
special structure. **It is also exactly what DeepSurv and DeepHit use**, so
you already have the architecture your dissertation needs.

But when the data has *structure*, we can wire that structure into the
network itself. This page is a conceptual tour of the three families you'll
hear about constantly, so the vocabulary holds no fear. You won't need them
for the papers — but you will need them in interviews.

The unifying idea: **each architecture builds a different assumption
(an "inductive bias") into its wiring.**
"""
)

st.header("1 · CNNs — for data where *position* matters, and patterns repeat")
st.markdown(
    """
Take a vibration signal from a machine: 40 readings over time. A fault shows
up as a characteristic *local shape* — a spike, a step — and that shape means
the same thing whether it appears early or late in the window.

A fully-connected layer would have to learn "what a spike looks like"
separately for every position. A **convolution** instead learns *one* small
pattern-detector (a **kernel** or filter) and **slides it across** the whole
signal. Two enormous wins: far fewer parameters, and the detector
automatically works anywhere in the signal (**translation invariance**).

Below is a real convolution: drag the kernel's position and watch the output
value at that point being computed from just the readings under the window.
"""
)

rng = np.random.default_rng(7)
signal = np.concatenate([
    rng.normal(0, 0.25, 14),
    np.array([0.3, 1.4, 4.2, 1.3, 0.2]),      # the "fault" spike
    rng.normal(0, 0.25, 12),
    np.array([0.5, 1.7, 1.0]),                # a smaller one
    rng.normal(0, 0.25, 6),
])
KERNELS = {
    "spike detector  [-1, 2, -1]": np.array([-1.0, 2.0, -1.0]),
    "smoother (averages)  [⅓, ⅓, ⅓]": np.array([1 / 3, 1 / 3, 1 / 3]),
    "edge / step detector  [-1, 0, 1]": np.array([-1.0, 0.0, 1.0]),
}
kname = st.selectbox("the kernel this neuron has learned", list(KERNELS))
kernel = KERNELS[kname]
pos = st.slider("slide the kernel along the signal", 0,
                len(signal) - len(kernel), 14)

window = signal[pos:pos + len(kernel)]
out_val = float(np.dot(window, kernel))
full_out = np.convolve(signal, kernel[::-1], mode="valid")  # the real thing

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(signal, color="#37474f", linewidth=1.5, marker="o",
             markersize=3)
axes[0].add_patch(Rectangle((pos - 0.4, signal.min() - 0.3), len(kernel) - 0.2,
                            signal.max() - signal.min() + 0.6,
                            facecolor="#ffe082", alpha=0.6, edgecolor="#f9a825",
                            linewidth=2))
for i, (idx, v, k) in enumerate(zip(range(pos, pos + len(kernel)), window,
                                    kernel)):
    axes[0].annotate(f"×{k:+.2f}", (idx, v), textcoords="offset points",
                     xytext=(0, 12), fontsize=8, ha="center", color="#e65100")
axes[0].set_ylabel("vibration")
axes[0].set_title("the raw sensor signal, with the kernel's window "
                  "highlighted", fontsize=10)
axes[0].grid(alpha=0.25)

axes[1].plot(range(len(full_out)), full_out, color="#1565c0", linewidth=1.5,
             marker="o", markersize=3)
axes[1].scatter([pos], [out_val], s=140, color="#e65100", zorder=5)
axes[1].axhline(0, color="#b0bec5", linewidth=0.8)
axes[1].set_xlabel("position along the signal")
axes[1].set_ylabel("kernel response")
axes[1].set_title(f"the convolution's output — at position {pos}: "
                  f"{' + '.join(f'({v:.2f})({k:+.2f})' for v, k in zip(window, kernel))}"
                  f" = {out_val:+.2f}", fontsize=9)
axes[1].grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

if kname.startswith("spike"):
    st.success(
        f"The spike detector fires hard (output {full_out.max():+.2f} at its "
        "peak) exactly where the fault spikes are, and stays near zero in the "
        "noise. **One kernel — three numbers — found the fault anywhere in "
        "the signal.** A real CNN learns dozens of such kernels from data "
        "rather than being handed them, then stacks layers so that later "
        "kernels detect patterns *of patterns*."
    )
else:
    st.info(
        "Try the **spike detector** to see a kernel that's actually tuned to "
        "this fault. Each kernel answers one question about the local shape "
        "of the signal — and a CNN learns which questions are worth asking."
    )
st.markdown(
    "Swap the 1-D signal for a 2-D grid of pixels and you have image CNNs; "
    "the kernel becomes a little square that slides in both directions, "
    "detecting edges, then textures, then shapes. Same mechanism."
)

st.header("2 · RNNs — for sequences, where *order* and memory matter")
st.markdown(
    """
Now suppose you want to predict failure from a machine's *history* of
readings, where the sequence can be any length and what matters is how
things evolved. A **recurrent** network walks through the sequence one
step at a time, carrying a **hidden state** — its memory of everything so
far:

$$h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)$$

Note what's reused: **the same weights at every time step**. That's the
inductive bias — "the rule for updating my understanding doesn't depend on
what o'clock it is."
"""
)
n_steps = st.slider("unroll the RNN over how many time steps?", 2, 6, 4)

fig, ax = plt.subplots(figsize=(10, 3.2))
for t in range(n_steps):
    x0 = 1.2 + t * 1.7
    ax.add_patch(Rectangle((x0 - 0.42, 1.1), 0.84, 0.85, facecolor="#fff3e0",
                           edgecolor="#e65100", linewidth=2))
    ax.text(x0, 1.52, f"h{t + 1}", ha="center", va="center", fontsize=11,
            fontweight="bold")
    ax.add_patch(Circle((x0, 0.35), 0.26, facecolor="#e3f2fd",
                        edgecolor="#1565c0", linewidth=1.6))
    ax.text(x0, 0.35, f"x{t + 1}", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((x0, 0.63), (x0, 1.05), arrowstyle="-|>",
                                 mutation_scale=12, color="#546e7a"))
    if t < n_steps - 1:
        ax.add_patch(FancyArrowPatch((x0 + 0.45, 1.52), (x0 + 1.24, 1.52),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#e65100", linewidth=2))
        ax.text(x0 + 0.85, 1.72, "same\nweights", ha="center", fontsize=6.5,
                color="#e65100")
    ax.text(x0, 0.02, f"reading\nat t={t + 1}", ha="center", fontsize=7,
            color="#90a4ae")
x_last = 1.2 + (n_steps - 1) * 1.7
ax.add_patch(FancyArrowPatch((x_last, 2.0), (x_last, 2.55), arrowstyle="-|>",
                             mutation_scale=13, color="#37474f"))
ax.add_patch(Circle((x_last, 2.85), 0.3, facecolor="#ffcdd2",
                    edgecolor="#37474f", linewidth=1.8))
ax.text(x_last, 2.85, "ŷ", ha="center", va="center", fontsize=11)
ax.text(x_last + 0.5, 2.85, "P(fails next month)", fontsize=8,
        color="#546e7a", va="center")
ax.set_xlim(0.2, 1.2 + n_steps * 1.7 + 2.2)
ax.set_ylim(-0.25, 3.3)
ax.axis("off")
ax.set_title("an RNN, unrolled: memory flows left to right through h",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
The problem with plain RNNs: to learn from something that happened 50 steps
ago, blame must be multiplied backwards through 50 tanh derivatives — and
you already know from the activations page what that does. **Vanishing
gradients**, so early events are forgotten. The fix was **LSTM** and **GRU**
cells, which add explicit "gates" that decide what to keep, forget, and
output, giving gradients a protected highway to flow along.

This is also where the **cricket** connection lives, if you want one: a
batter's dismissal risk over an innings is a sequence — balls faced, runs
scored, bowler changes — and an RNN is a natural way to carry "how settled
is this batter *right now*" forward through the innings.
"""
)

st.header("3 · Transformers — look at everything at once, and weigh it")
st.markdown(
    """
RNNs read strictly left-to-right, which makes them slow (no parallelism) and
forgetful. **Transformers** threw out recurrence entirely for **attention**:
every position looks at *every other* position simultaneously and decides how
much each one matters to it.

For each pair of positions, attention computes a **weight** — "how much
should position *i* pay attention to position *j*?" — and each position's new
representation becomes a weighted blend of all the others. Below is a real
(hand-set, for illustration) attention matrix over one over of a cricket
innings, showing what the model attends to when judging **dismissal risk on
ball 6**:
"""
)
balls = ["ball 1\ndot", "ball 2\nedged 4", "ball 3\ndot",
         "ball 4\nbeaten", "ball 5\ndot", "ball 6\n?"]
raw_att = np.array([
    [3.0, 0.4, 0.5, 0.3, 0.4, 0.2],
    [0.4, 3.0, 0.6, 0.9, 0.5, 0.3],
    [0.5, 0.6, 3.0, 1.2, 0.8, 0.4],
    [0.3, 0.9, 1.2, 3.0, 1.4, 0.6],
    [0.4, 0.5, 0.8, 1.4, 3.0, 0.9],
    [0.3, 1.6, 0.9, 2.4, 2.0, 3.0],   # ball 6 attends to the danger signs
])
att = np.exp(raw_att) / np.exp(raw_att).sum(axis=1, keepdims=True)  # softmax!

fig, ax = plt.subplots(figsize=(7, 5.4))
im = ax.imshow(att, cmap="Purples", vmin=0, vmax=att.max())
fig.colorbar(im, ax=ax, label="attention weight (each row sums to 1)")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{att[i, j]:.2f}", ha="center", va="center",
                fontsize=8,
                color="white" if att[i, j] > att.max() * 0.55 else "#37474f")
ax.set_xticks(range(6))
ax.set_xticklabels(balls, fontsize=7)
ax.set_yticks(range(6))
ax.set_yticklabels(balls, fontsize=7)
ax.set_xlabel("…attends to these balls")
ax.set_ylabel("this ball…")
ax.add_patch(Rectangle((-0.5, 4.5), 6, 1, fill=False, edgecolor="#e65100",
                       linewidth=3))
ax.set_title("a self-attention matrix (illustrative weights, real softmax)",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
Read the highlighted bottom row: when assessing ball 6, this model puts
{att[5, 3]:.0%} of its attention on **ball 4 (beaten)** and {att[5, 4]:.0%}
on ball 5 — the recent signs of trouble — and only {att[5, 0]:.0%} on the
irrelevant first ball. It reaches ball 4 *directly*, in one hop, with no
memory to fade. That's the transformer's superpower, and why it displaced
RNNs.

Two things you already know, hiding in here:
- Those weights come from **softmax** (activations page) — each row is a
  probability distribution over "where to look".
- The whole thing is matrix multiplications — which is why transformers
  parallelise beautifully on GPUs, and why they scaled into the large
  language models you use every day (including me).
"""
)

st.header("4 · So which do I use?")
st.markdown(
    """
| Your data looks like… | Reach for | Because |
|---|---|---|
| a flat list of features (sensors, patient covariates, cricket stats) | **fully-connected** (`nn.Linear`) | no spatial or sequential structure to exploit — **this is DeepSurv & DeepHit** |
| a grid where local patterns repeat (images, spectrograms, 1-D signals) | **CNN** | weight sharing + translation invariance |
| a sequence where order matters and length varies | **RNN / LSTM / GRU** | carries a memory forward |
| a sequence where *long-range* relationships matter | **Transformer** | every position reaches every other in one hop |
| tabular data, honestly | **gradient boosting** (Section 3!) | still beats deep nets on most tabular problems — don't reach for a neural net out of fashion |

**That last row is not a joke, and it's the most professionally useful
sentence in this section.** For a plain table of features, XGBoost is usually
the stronger and cheaper choice. Neural networks earn their keep when the
data has structure to exploit — *or* when you need something a tree can't
give you, such as **a custom loss function**.

And that is precisely the doorway into your dissertation. DeepSurv and
DeepHit are ordinary fully-connected networks — the ones you built on the
last four pages. What makes them powerful isn't the architecture; **it's the
loss function**, engineered to handle censored time-to-failure data. That
loss is where Sections 6 and 7 begin.
"""
)

show_example(
    '''import torch
import torch.nn as nn

# The three families, declared. Note how little the code differs.
mlp = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
cnn = nn.Sequential(nn.Conv1d(1, 8, kernel_size=3), nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(8, 1))
rnn = nn.GRU(input_size=1, hidden_size=8, batch_first=True)

signal = torch.randn(4, 1, 40)          # 4 machines, 1 channel, 40 readings
print("CNN output shape:", cnn(signal).shape)

seq = torch.randn(4, 40, 1)             # 4 machines, 40 time steps, 1 feature
out, h = rnn(seq)
print("RNN all-steps shape:", out.shape, " final memory:", h.shape)

flat = torch.randn(4, 12)               # 4 machines, 12 summary features
print("MLP output shape:", mlp(flat).shape)

for name, m in [("MLP", mlp), ("CNN", cnn), ("GRU", rnn)]:
    print(f"{name} parameters:", sum(p.numel() for p in m.parameters()))''',
    """
- `nn.Conv1d(1, 8, kernel_size=3)` — 8 learned kernels, each 3 wide (like the sliding window above, but the numbers are *learned*). `1` is the number of input channels.
- `nn.AdaptiveMaxPool1d(1)` — "did this kernel fire *anywhere*?" — squashing the position axis away, which is what makes the detector position-independent.
- `nn.GRU(...)` — a gated RNN. It returns both the output at every step and the final hidden state (the memory). `batch_first=True` means the input is shaped `(batch, time, features)`.
- The shapes are the lesson: the CNN and RNN consume *structured* inputs (a 40-long signal), the MLP consumes a flat 12-feature summary.
- The parameter counts show the CNN's efficiency: its 8 kernels of 3 weights each cover a 40-long signal with a fraction of the parameters a fully-connected layer would need.
""",
)

guided_sandbox(
    key="n7",
    steps="""
1. **Step 1** — apply the spike-detector kernel `[-1, 2, -1]` to the signal
   by hand: loop over positions and compute `np.dot(signal[i:i+3], kernel)`.
   Collect the outputs into a list.
2. **Step 2** — check your work against NumPy:
   `np.convolve(signal, kernel[::-1], mode="valid")` (the `[::-1]` is because
   `np.convolve` flips the kernel by mathematical convention). Print whether
   they match with `np.allclose`.
3. **Step 3** — find where the detector fires hardest with `np.argmax`, and
   confirm it lands on the fault spike planted at index 14–18.
4. **Step 4 (stretch)** — build a real `nn.Conv1d(1, 1, 3)` layer, force its
   weight to your kernel (`layer.weight.data = torch.tensor(kernel).reshape(
   1, 1, 3)`, `layer.bias.data.zero_()`), push the signal through, and check
   PyTorch agrees with your hand-rolled convolution.
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(7)
signal = np.concatenate([
    rng.normal(0, 0.25, 14),
    np.array([0.3, 1.4, 4.2, 1.3, 0.2]),      # the fault spike (index 14-18)
    rng.normal(0, 0.25, 12),
    np.array([0.5, 1.7, 1.0]),                # a smaller blip
    rng.normal(0, 0.25, 6),
])
kernel = np.array([-1.0, 2.0, -1.0])
print("signal length:", len(signal))

# Step 1: slide the kernel by hand, collect the outputs


# Step 2: compare with np.convolve(signal, kernel[::-1], mode="valid")


# Step 3: np.argmax - where does it fire hardest?


# Step 4 (stretch): the same thing with a real nn.Conv1d layer
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(7)
signal = np.concatenate([
    rng.normal(0, 0.25, 14),
    np.array([0.3, 1.4, 4.2, 1.3, 0.2]),
    rng.normal(0, 0.25, 12),
    np.array([0.5, 1.7, 1.0]),
    rng.normal(0, 0.25, 6),
])
kernel = np.array([-1.0, 2.0, -1.0])

mine = []
for i in range(len(signal) - len(kernel) + 1):
    mine.append(np.dot(signal[i:i + len(kernel)], kernel))
mine = np.array(mine)
print("my first 5 outputs:", mine[:5].round(3))

theirs = np.convolve(signal, kernel[::-1], mode="valid")
print("numpy agrees:", np.allclose(mine, theirs))

peak = int(np.argmax(mine))
print(f"fires hardest at position {peak} (value {mine[peak]:.2f})")
print("the fault spike was planted at index 14-18 ->",
      "found it!" if 13 <= peak <= 18 else "somewhere else")

layer = nn.Conv1d(1, 1, 3, bias=False)
layer.weight.data = torch.tensor(kernel, dtype=torch.float32).reshape(1, 1, 3)
x = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, -1)
with torch.no_grad():
    torch_out = layer(x).squeeze().numpy()
print("PyTorch Conv1d agrees:", np.allclose(torch_out, mine, atol=1e-5))''',
)
