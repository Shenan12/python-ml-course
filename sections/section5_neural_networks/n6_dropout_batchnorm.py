import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from utils.sandbox import guided_sandbox, show_example

st.title("🎛️ Dropout & Batch Normalization")
st.markdown(
    """
Big networks overfit — the U-curve from Section 2 does not spare them. This
page covers the two techniques that appear in nearly every modern
architecture (and both appear in the DeepSurv paper's hyperparameter search,
so this is direct dissertation groundwork).

- **Dropout** — during *training only*, randomly switch off a fraction of
  neurons on every single forward pass. It sounds like sabotage. It's the
  most effective regularizer in deep learning.
- **Batch normalization** — standardize each layer's outputs (subtract the
  batch mean, divide by the batch std) *inside* the network, so every layer
  receives well-behaved inputs no matter how the previous layers drift.

Our data is a deliberately hard one for this demo: a small, noisy sample of
**equipment sensor readings** — only 40 training units, 12 input features of
which just 3 actually matter (the rest are noise sensors), and a network far
too big for the job. A perfect overfitting trap.
"""
)


@st.cache_data(show_spinner="Generating the noisy sensor data…")
def make_data():
    rng = np.random.default_rng(4)
    n_tr, n_va, d = 40, 300, 12
    Xtr = rng.normal(0, 1, (n_tr, d))
    Xva = rng.normal(0, 1, (n_va, d))
    w_true = np.zeros(d)
    w_true[:3] = [1.8, -1.6, 1.4]           # only 3 sensors matter
    def label(X):
        z = X @ w_true
        p = 1 / (1 + np.exp(-z))
        return (rng.uniform(0, 1, len(X)) < p).astype(np.float32)
    return (Xtr.astype(np.float32), label(Xtr),
            Xva.astype(np.float32), label(Xva))


Xtr, ytr, Xva, yva = make_data()
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)


@st.cache_data(show_spinner="Training…")
def train(use_dropout, p_drop, use_bn, epochs=300, seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(12, 128)]
    if use_bn:
        layers.append(nn.BatchNorm1d(128))
    layers.append(nn.ReLU())
    if use_dropout:
        layers.append(nn.Dropout(p_drop))
    layers += [nn.Linear(128, 64)]
    if use_bn:
        layers.append(nn.BatchNorm1d(64))
    layers.append(nn.ReLU())
    if use_dropout:
        layers.append(nn.Dropout(p_drop))
    layers.append(nn.Linear(64, 1))
    model = nn.Sequential(*layers)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    tr_hist, va_hist = [], []
    for _ in range(epochs):
        model.train()                       # dropout ON, BN uses batch stats
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t).squeeze(1), ytr_t)
        loss.backward()
        opt.step()
        model.eval()                        # dropout OFF, BN uses running stats
        with torch.no_grad():
            tr_hist.append(loss_fn(model(Xtr_t).squeeze(1), ytr_t).item())
            va_hist.append(loss_fn(model(Xva_t).squeeze(1), yva_t).item())
    model.eval()
    with torch.no_grad():
        acc = (((model(Xva_t).squeeze(1) > 0).float() == yva_t)
               .float().mean().item())
    return tr_hist, va_hist, acc


st.header("1 · Watch the overfitting, then cure it")
c1, c2, c3 = st.columns(3)
use_dropout = c1.toggle("Dropout", value=False)
p_drop = c2.select_slider("dropout probability p", [0.1, 0.3, 0.5, 0.8],
                          value=0.5, disabled=not use_dropout)
use_bn = c3.toggle("Batch normalization", value=False)

tr_hist, va_hist, acc = train(use_dropout, p_drop, use_bn)
base_tr, base_va, base_acc = train(False, 0.5, False)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tr_hist, color="#1565c0", linewidth=2, label="training loss")
ax.plot(va_hist, color="#e65100", linewidth=2, label="validation loss")
if use_dropout or use_bn:
    ax.plot(base_va, color="#bdbdbd", linewidth=1.4, linestyle="--",
            label="validation loss with NO defences (for comparison)")
best_epoch = int(np.argmin(va_hist))
ax.axvline(best_epoch, color="#2e7d32", linestyle=":",
           label=f"best epoch ({best_epoch}) — where early stopping would stop")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.set_title("training vs validation loss", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("final training loss", f"{tr_hist[-1]:.4f}")
c2.metric("best validation loss", f"{min(va_hist):.4f}",
          delta=f"{min(va_hist) - min(base_va):+.4f} vs no defences",
          delta_color="inverse")
c3.metric("validation accuracy", f"{acc:.0%}",
          delta=f"{(acc - base_acc) * 100:+.1f} pts vs no defences")

if not use_dropout and not use_bn:
    st.error(
        f"**Textbook overfitting.** The training loss dives toward zero — "
        f"the network is memorising all 40 units, noise sensors and all — "
        f"while the validation loss bottoms out around epoch {best_epoch} "
        "and then climbs steadily. Every epoch after that green line makes "
        "the model *worse* on unseen equipment. Now switch on a defence."
    )
else:
    which = " + ".join([n for n, on in [("Dropout", use_dropout),
                                        ("BatchNorm", use_bn)] if on])
    st.success(
        f"**{which} on.** Compare the orange curve against the grey dashed "
        "line (the undefended run): the validation loss stays lower and its "
        "climb is flattened. The training loss is now *worse* — and that is "
        "the point. We deliberately handicapped the network's ability to "
        "memorise, and bought generalisation with it."
    )

st.header("2 · What dropout actually does, per forward pass")
st.markdown(
    "Each square is a neuron in a layer of 24. On every training pass, "
    "PyTorch draws a fresh random mask — the greyed-out neurons are **zeroed "
    "for that pass only**. Below is a real `nn.Dropout` applied to a vector "
    "of ones, three separate times:"
)
p_show = st.select_slider("dropout probability p (for this illustration)",
                          [0.1, 0.3, 0.5, 0.8], value=0.5)
torch.manual_seed(int(p_show * 100))
drop = nn.Dropout(p_show)
drop.train()
passes = [drop(torch.ones(24)).numpy() for _ in range(3)]

fig, axes = plt.subplots(3, 1, figsize=(9.5, 2.9))
for k, (ax, vals) in enumerate(zip(axes, passes)):
    for i, v in enumerate(vals):
        alive = v > 0
        ax.add_patch(plt.Rectangle((i, 0), 0.88, 1,
                                   facecolor="#66bb6a" if alive else "#eceff1",
                                   edgecolor="#546e7a"))
        if alive:
            ax.text(i + 0.44, 0.5, f"{v:.1f}", ha="center", va="center",
                    fontsize=7, color="white")
        else:
            ax.text(i + 0.44, 0.5, "✕", ha="center", va="center", fontsize=9,
                    color="#b0bec5")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"pass {k + 1}", fontsize=8, rotation=0, labelpad=24,
                  va="center")
    ax.set_xticks([])
    ax.set_yticks([])
st.pyplot(fig)
plt.close(fig)
alive_frac = np.mean([(pp > 0).mean() for pp in passes])
st.markdown(
    f"""
Three passes, three different random masks — roughly {alive_frac:.0%} of
neurons survived each time (p = {p_show}, so ≈ {1 - p_show:.0%} expected).
Two details that matter:

- **The survivors are scaled up** (notice they show {1 / (1 - p_show):.2f},
  not 1.0). PyTorch divides by (1 − p) so the layer's *total* signal stays
  the same on average — that way the network sees consistent magnitudes
  whether dropout is on or off.
- **At evaluation time dropout is switched off entirely** — that's what
  `model.eval()` does, and forgetting it is a genuinely common bug that
  makes your test predictions randomly wobble.

**Why sabotage helps:** a neuron cannot rely on any *particular* other neuron
being present, because its neighbours keep vanishing. So the network can't
build fragile, co-dependent chains ("neuron 7 only works if neuron 12 fires")
— it's forced to spread the representation redundantly across many neurons.
It also behaves like training a huge *ensemble* of thinned networks that share
weights, and then averaging them at test time — the bagging idea from
Section 3, smuggled inside a single model.
"""
)

st.header("3 · What batch normalization actually does")
st.markdown(
    "Left: the raw outputs of a hidden layer (badly scaled and off-centre — "
    "this happens naturally as layers drift during training). Right: the same "
    "values after a real `nn.BatchNorm1d`. Both computed live:"
)
torch.manual_seed(1)
raw = torch.randn(200, 3) * torch.tensor([4.0, 0.3, 1.5]) + torch.tensor(
    [6.0, -2.0, 0.5])
bn = nn.BatchNorm1d(3)
bn.train()
normed = bn(raw).detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
for ax, data, title in [(axes[0], raw.numpy(), "before BatchNorm"),
                        (axes[1], normed, "after BatchNorm")]:
    for j, colour in enumerate(["#1565c0", "#e65100", "#2e7d32"]):
        ax.hist(data[:, j], bins=30, alpha=0.55, color=colour,
                label=f"neuron {j + 1}")
    ax.set_title(f"{title}\nmeans: {data.mean(axis=0).round(2)}  "
                 f"stds: {data.std(axis=0).round(2)}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Before: three neurons living on wildly different scales, none centred on
zero. After: all three centred at ≈0 with std ≈1. Why the next layer cares:

- **It keeps activations in the sensitive zone.** A tanh or sigmoid fed
  values of ±6 is saturated — flat derivative, no learning (the vanishing
  gradient again). Normalized inputs sit where the activation actually
  responds.
- **It rounds out the loss surface** — the same reason we standardized inputs
  in Section 2, but applied *at every layer*, continuously, during training.
  Rounder bowls are easier to descend, so you can use larger learning rates
  and train faster.
- **It has a mild regularizing side-effect**, because each example's
  normalization depends on the random other examples in its mini-batch —
  a little noise, in the dropout spirit.

BatchNorm also learns two parameters per neuron (a scale γ and a shift β), so
it can *undo* the normalization if that turns out to be what the network
wants. And like dropout, it behaves differently in `train()` vs `eval()` mode:
at evaluation it uses running averages collected during training, not the
current batch's statistics (you don't want your prediction for one pump to
depend on which other pumps happen to be in the batch).
"""
)

st.header("4 · In code")
show_example(
    '''import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.BatchNorm1d(64),     # normalize this layer's 64 outputs
    nn.ReLU(),
    nn.Dropout(0.5),        # then randomly kill half of them (training only)
    nn.Linear(64, 1),
)
print(model)

x = torch.randn(8, 12)      # a mini-batch of 8 units

model.train()               # TRAINING mode
out1 = model(x)
out2 = model(x)
print("train mode, same input twice - identical?",
      torch.allclose(out1, out2), "(dropout randomises each pass)")

model.eval()                # EVALUATION mode
with torch.no_grad():
    out3 = model(x)
    out4 = model(x)
print("eval mode, same input twice - identical?",
      torch.allclose(out3, out4), "(dropout off, BN uses running stats)")''',
    """
- The layer order `Linear → BatchNorm → ReLU → Dropout` is the conventional stack. (You'll see BN placed after the activation too; both are defensible and the debate is mostly empirical.)
- `model.train()` / `model.eval()` — these **flip the behaviour of dropout and BatchNorm**, and nothing else. The printed comparison proves it: in train mode the same input gives *different* outputs each pass (random dropout masks), in eval mode it is perfectly deterministic. Forgetting `model.eval()` before evaluating is the bug this snippet exists to inoculate you against.
- `torch.no_grad()` — no gradient graph needed when you're only predicting.
- Note `nn.BatchNorm1d` needs a batch with more than one example in train mode (it needs a mean and a std!) — feeding it a single row throws an error.
""",
)

guided_sandbox(
    key="n6",
    steps="""
1. **Step 1** — build a model with dropout:
   `nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.5),
   nn.Linear(64, 1))`.
2. **Step 2** — prove dropout is random in training mode: call `model.train()`,
   push `Xtr_t` through twice, and check whether the two outputs are equal
   (`torch.allclose`). Then call `model.eval()` and repeat. Print both answers.
3. **Step 3** — train it for 300 epochs with `Adam(lr=0.01)` and
   `BCEWithLogitsLoss` (remember `model.train()` before the step and
   `model.eval()` before evaluating), recording validation loss each epoch.
   Print the best validation loss.
4. **Step 4 (stretch)** — train the same architecture **without** dropout
   and print both best validation losses side by side. Which generalises
   better on this deliberately tiny, noisy dataset?
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(4)
Xtr = rng.normal(0, 1, (40, 12)).astype(np.float32)
Xva = rng.normal(0, 1, (300, 12)).astype(np.float32)
w_true = np.zeros(12); w_true[:3] = [1.8, -1.6, 1.4]
def label(X):
    p = 1 / (1 + np.exp(-(X @ w_true)))
    return (rng.uniform(0, 1, len(X)) < p).astype(np.float32)
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(label(Xtr))
Xva_t, yva_t = torch.tensor(Xva), torch.tensor(label(Xva))
torch.manual_seed(0)
print("40 training units, 12 features (only 3 informative)")

# Step 1: build the model with Dropout(0.5)


# Step 2: show train() is random and eval() is deterministic


# Step 3: train 300 epochs, track validation loss, print the best


# Step 4 (stretch): the same model without dropout - compare
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(4)
Xtr = rng.normal(0, 1, (40, 12)).astype(np.float32)
Xva = rng.normal(0, 1, (300, 12)).astype(np.float32)
w_true = np.zeros(12); w_true[:3] = [1.8, -1.6, 1.4]
def label(X):
    p = 1 / (1 + np.exp(-(X @ w_true)))
    return (rng.uniform(0, 1, len(X)) < p).astype(np.float32)
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(label(Xtr))
Xva_t, yva_t = torch.tensor(Xva), torch.tensor(label(Xva))

def build(dropout):
    torch.manual_seed(0)
    layers = [nn.Linear(12, 64), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)

model = build(True)

model.train()
a, b = model(Xtr_t), model(Xtr_t)
print("train mode identical?", torch.allclose(a, b))
model.eval()
with torch.no_grad():
    c, d = model(Xtr_t), model(Xtr_t)
print("eval mode identical? ", torch.allclose(c, d))

def run(dropout):
    model = build(dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    best = float("inf")
    for _ in range(300):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t).squeeze(1), ytr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva_t).squeeze(1), yva_t).item()
        best = min(best, va)
    return best

with_do = run(True)
without = run(False)
print(f"best val loss WITH dropout:    {with_do:.4f}")
print(f"best val loss WITHOUT dropout: {without:.4f}")
print("lower is better")''',
)
