import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from utils.mockdata import pumps_xor
from utils.sandbox import guided_sandbox, show_example

st.title("🔁 The Training Loop")
st.markdown(
    """
You now have every ingredient: a forward pass, a loss, and backprop to get
gradients. The **training loop** is just those three in a `for` loop, plus
an optimiser that applies the update. Every neural network ever trained —
including DeepSurv and DeepHit — runs this loop:

```
for each epoch:                     # one epoch = one sweep through the data
    for each mini-batch of data:
        1. forward   → predictions
        2. loss      → how wrong?
        3. backward  → gradients (backprop)
        4. step      → optimiser nudges every weight downhill
        5. zero the gradients (they ACCUMULATE otherwise — classic bug)
```

This page is also where we stop hand-rolling matrices and let **PyTorch**
do it — the same library DeepSurv and DeepHit are built on (via `pycox`).
"""
)

X, y = pumps_xor()
Xs = ((X - 5.0) / 2.5).astype(np.float32)
ys = y.astype(np.float32)
n_train = 140
Xtr, Xva = torch.tensor(Xs[:n_train]), torch.tensor(Xs[n_train:])
ytr, yva = torch.tensor(ys[:n_train]), torch.tensor(ys[n_train:])


@st.cache_data(show_spinner="Training networks…")
def train(lr, hidden, epochs, batch_size, optimiser, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if optimiser == "SGD"
           else torch.optim.Adam(model.parameters(), lr=lr))
    tr_hist, va_hist = [], []
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()                             # 5 (done first)
            logits = model(Xtr[idx]).squeeze(1)         # 1 forward
            loss = loss_fn(logits, ytr[idx])            # 2 loss
            loss.backward()                             # 3 backward
            opt.step()                                  # 4 step
        with torch.no_grad():
            tr = loss_fn(model(Xtr).squeeze(1), ytr).item()
            va = loss_fn(model(Xva).squeeze(1), yva).item()
            acc = ((model(Xva).squeeze(1) > 0).float() == yva).float().mean()
        tr_hist.append(tr)
        va_hist.append(va)
    return tr_hist, va_hist, float(acc), model


st.header("1 · Turn the knobs, watch the loss curves")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
lr = c1.select_slider("learning rate", [0.001, 0.01, 0.1, 0.5, 5.0],
                      value=0.1)
hidden = c2.select_slider("hidden neurons", [1, 2, 4, 16, 64], value=4)
batch_size = c3.select_slider("mini-batch size", [1, 8, 32, 140], value=32)
optimiser = c4.radio("optimiser", ["SGD", "Adam"], horizontal=True)
epochs = 120

tr_hist, va_hist, acc, model = train(lr, hidden, epochs, batch_size,
                                     optimiser)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(tr_hist, color="#1565c0", linewidth=1.8, label="training loss")
axes[0].plot(va_hist, color="#e65100", linewidth=1.8, label="validation loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss (binary cross-entropy)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_title(f"the learning curve — final val accuracy {acc:.0%}",
                  fontsize=10)

gx, gy = np.meshgrid(np.linspace(0, 10, 120), np.linspace(0, 10, 120))
grid = torch.tensor(np.column_stack([(gx.ravel() - 5) / 2.5,
                                     (gy.ravel() - 5) / 2.5]),
                    dtype=torch.float32)
with torch.no_grad():
    zz = (model(grid).squeeze(1) > 0).numpy().reshape(gx.shape)
axes[1].contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
                 colors=["#c8e6c9", "#ffcdd2"], alpha=0.75)
for cls, colour in [(0, "#2e7d32"), (1, "#c62828")]:
    m = y == cls
    axes[1].scatter(X[m, 0], X[m, 1], c=colour, s=18, edgecolor="white")
axes[1].set_xlabel("vibration")
axes[1].set_ylabel("temperature")
axes[1].set_aspect("equal")
axes[1].set_title("the boundary this run learned", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("final training loss", f"{tr_hist[-1]:.4f}")
c2.metric("final validation loss", f"{va_hist[-1]:.4f}")
c3.metric("validation accuracy", f"{acc:.0%}")

if lr >= 5.0:
    st.error(
        "**Learning rate far too high.** The loss curve is thrashing or "
        "stuck — steps overshoot the valley every time. Exactly the "
        "divergence you saw walking down the loss surface in Section 2, now "
        "with 9+ parameters instead of 2."
    )
elif lr <= 0.001 and optimiser == "SGD":
    st.warning(
        "**Learning rate far too low.** The curve is still creeping "
        "downhill when the epochs run out — the network is *learning*, just "
        "far too slowly. More epochs would rescue it; a bigger step would "
        "be cheaper."
    )
elif hidden == 1:
    st.warning(
        "**One hidden neuron is not enough.** With a single hidden unit the "
        "network can still only carve a straight-ish boundary — the XOR "
        "pattern needs at least two hidden neurons (the forward-pass page "
        "showed exactly why). This is underfitting by architecture."
    )
else:
    st.success(
        f"A healthy run: both curves fall and flatten together, ending at "
        f"{acc:.0%} validation accuracy."
    )

st.markdown(
    """
Things worth discovering with the knobs (each is a real experiment — the
networks retrain live):

- **batch size 1 vs 140.** Size 1 (pure stochastic gradient descent) makes
  the curve *noisy* — each step is based on a single pump's opinion — but it
  takes many more steps per epoch. Size 140 (the whole training set = "full
  batch") gives a beautifully smooth curve and slower progress per epoch.
  Mini-batches (8–32) are the practical compromise everyone uses: enough
  examples to average out noise, small enough to step often. That noise
  isn't purely a nuisance, either — it helps the optimiser rattle out of bad
  local minima.
- **SGD vs Adam** at a low learning rate like 0.001. SGD crawls; **Adam**
  romps ahead. Adam adapts a per-parameter step size on the fly (it tracks
  each gradient's recent average and variance). It's the default in modern
  deep learning, and — file this away — **it's the optimiser the DeepSurv
  paper uses**, alongside learning-rate decay.
- **hidden = 64 vs 2.** More neurons = more flexibility. Watch whether the
  validation curve starts drifting *up* while training loss keeps falling:
  that's overfitting, the Section 2 U-curve appearing yet again, and the
  cue for the next page's defences.
"""
)

st.header("2 · The loop in code — PyTorch, at last")
show_example(
    '''import numpy as np
import torch
import torch.nn as nn
from utils.mockdata import pumps_xor

X, y = pumps_xor()
Xt = torch.tensor(((X - 5.0) / 2.5), dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)

torch.manual_seed(0)
model = nn.Sequential(       # the 2-4-1 network, declared not hand-coded
    nn.Linear(2, 4),         # 2 inputs -> 4 hidden neurons (weights + bias)
    nn.Tanh(),               # the squash
    nn.Linear(4, 1),         # 4 hidden -> 1 output score (a "logit")
)
loss_fn = nn.BCEWithLogitsLoss()          # sigmoid + log-loss, fused
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    opt.zero_grad()                       # clear last round's gradients
    logits = model(Xt).squeeze(1)         # 1. forward
    loss = loss_fn(logits, yt)            # 2. loss
    loss.backward()                       # 3. backward (autograd does it)
    opt.step()                            # 4. update every weight
    if epoch % 50 == 0:
        acc = ((logits > 0).float() == yt).float().mean()
        print(f"epoch {epoch:3d}: loss {loss.item():.4f}  acc {acc:.0%}")

with torch.no_grad():
    final_acc = ((model(Xt).squeeze(1) > 0).float() == yt).float().mean()
print("final accuracy:", f"{final_acc:.0%}")
print("parameter count:", sum(p.numel() for p in model.parameters()))''',
    """
- `nn.Sequential(...)` — declares the architecture as a stack of layers. `nn.Linear(2, 4)` **is** the `x @ W + b` you hand-coded: it creates a 2×4 weight matrix and a 4-vector bias, and initialises them randomly for you.
- `nn.BCEWithLogitsLoss()` — binary cross-entropy (our log-loss) that expects raw scores and applies the sigmoid *internally*. That's why the model's last layer has no activation: the loss fuses them, which is both numerically safer and exactly the `p − y` cancellation you proved on the backprop page.
- `torch.optim.Adam(model.parameters(), lr=0.05)` — hand the optimiser every weight in the model; it will update them all.
- `opt.zero_grad()` — **PyTorch accumulates gradients by default**; forget this line and every step adds to the last step's gradients, and training silently goes haywire. It is *the* classic beginner bug.
- `loss.backward()` — autograd walks the graph of operations backwards and fills in `.grad` for every parameter. This one line replaces the entire hand-derived backward pass from the previous page — and you now know precisely what it's doing under the hood.
- `opt.step()` — applies `w ← w − lr · gradient` (with Adam's adaptive twist) to all 17 parameters.
- `with torch.no_grad():` — "just predict, don't build a gradient graph": faster, and standard for evaluation.
""",
)

guided_sandbox(
    key="n5",
    steps="""
1. **Step 1** — build the model: `nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
   nn.Linear(8, 1))`. Print it, and print its parameter count with
   `sum(p.numel() for p in model.parameters())`.
2. **Step 2** — create `loss_fn = nn.BCEWithLogitsLoss()` and
   `opt = torch.optim.Adam(model.parameters(), lr=0.05)`.
3. **Step 3** — write the loop for 300 epochs with the five steps in the
   right order (`zero_grad`, forward, loss, `backward`, `step`). Print the
   loss and training accuracy every 100 epochs.
4. **Step 4 (stretch)** — deliberately **delete the `opt.zero_grad()` line**
   and re-run. The gradients now pile up from every previous step, and
   training is visibly crippled — the loss ends up *hundreds of times*
   worse. Note it doesn't crash or throw an error, which is exactly what
   makes this the most common silent bug in PyTorch. Then put the line back.
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn
from utils.mockdata import pumps_xor

X, y = pumps_xor()
Xt = torch.tensor((X - 5.0) / 2.5, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)
torch.manual_seed(0)
print("data ready:", Xt.shape, yt.shape)

# Step 1: build the model, print it and its parameter count


# Step 2: loss function and optimiser


# Step 3: the training loop (300 epochs), printing every 100


# Step 4 (stretch): remove opt.zero_grad() and watch it break
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn
from utils.mockdata import pumps_xor

X, y = pumps_xor()
Xt = torch.tensor((X - 5.0) / 2.5, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)
torch.manual_seed(0)

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(300):
    opt.zero_grad()
    logits = model(Xt).squeeze(1)
    loss = loss_fn(logits, yt)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        acc = ((logits > 0).float() == yt).float().mean()
        print(f"epoch {epoch:3d}: loss {loss.item():.4f}  acc {acc:.0%}")

with torch.no_grad():
    acc = ((model(Xt).squeeze(1) > 0).float() == yt).float().mean()
print(f"final accuracy: {acc:.0%}")

print("\\nnow WITHOUT zero_grad (the classic bug):")
torch.manual_seed(0)
bad = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
opt2 = torch.optim.Adam(bad.parameters(), lr=0.05)
for epoch in range(300):
    logits = bad(Xt).squeeze(1)          # no zero_grad!
    loss = loss_fn(logits, yt)
    loss.backward()
    opt2.step()
    if epoch % 100 == 0:
        print(f"  epoch {epoch:3d}: loss {loss.item():.4f}")
with torch.no_grad():
    bad_acc = ((bad(Xt).squeeze(1) > 0).float() == yt).float().mean()
print(f"  accuracy without zero_grad: {bad_acc:.0%} (gradients piled up)")''',
)
