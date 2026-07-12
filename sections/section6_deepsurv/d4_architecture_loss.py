import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from sections.section6_deepsurv import _experiments as experiments
from utils.mockdata import machines
from utils.sandbox import guided_sandbox, show_example

st.title("🧬 DeepSurv: Architecture & Loss Function")
st.markdown(
    r"""
Here is the whole model, in one substitution. The Cox model says:

$$h(t \mid x) = h_0(t) \cdot \exp(\underbrace{\beta^\top x}_{\text{linear}})$$

DeepSurv says:

$$h(t \mid x) = h_0(t) \cdot \exp(\underbrace{\hat h_\theta(x)}_{\text{a neural network}})$$

That's it. **Everything else is inherited from Cox**: still a proportional
hazards model, still no baseline hazard estimated, still fitted by maximising
the partial likelihood. The network is a *drop-in replacement for the linear
predictor*.

The loss, exactly as it appears in Katzman et al. (2018) — the average
negative log partial likelihood plus an L2 penalty:

$$\ell(\theta) := -\frac{1}{N_{E=1}}\sum_{i:\,E_i = 1} \left[\hat h_\theta(x_i) - \log\!\!\sum_{j \in \mathcal{R}(T_i)}\!\! e^{\hat h_\theta(x_j)}\right] \;+\; \lambda \cdot \|\theta\|_2^2$$

Every symbol should now be familiar:
- the bracket is **exactly Cox's partial likelihood term** from two pages ago,
  with $\beta^\top x_i$ swapped for $\hat h_\theta(x_i)$;
- the minus sign turns "maximise a likelihood" into "minimise a loss", so
  gradient *descent* applies (statistician's habit: MLE = minimising negative
  log-likelihood);
- $\lambda \|\theta\|_2^2$ is **ridge/L2 regularization** — the exact penalty
  you built on the Section 2 regularization page, now applied to the network's
  weights.

**The output layer is a single neuron with no activation and no bias.** Its
raw value *is* $\hat h_\theta(x)$: the log-risk. No sigmoid, no softmax —
because the partial likelihood does the normalising itself.
"""
)

st.header("1 · The architecture, as the paper describes it")
st.markdown(
    """
Per Katzman et al., DeepSurv is a multi-layer perceptron with a **single
output node** estimating $\\hat h_\\theta(x)$, using **SELU or ReLU**
activations (with **batch normalization** in the ReLU variants), **dropout**,
and **weight decay** (L2). Depth and width are hyperparameters, searched per
dataset: across their experiments the paper uses roughly **1–3 hidden
layers** with about **4–48 nodes** per layer, dropout in the range **~0.1–0.7**.

Note the *smallness*. These are not large networks — a few thousand
parameters at most. That is a deliberate response to the datasets being small
(1,638 units for WHAS) and heavily censored.
"""
)
c1, c2, c3 = st.columns(3)
n_layers = c1.slider("hidden layers", 1, 3, 2)
n_nodes = c2.select_slider("nodes per layer", [4, 8, 16, 32, 48], value=32)
act = c3.radio("activation", ["ReLU + BatchNorm", "SELU"], horizontal=False)

layers = []
in_dim = 10
for _ in range(n_layers):
    layers.append(nn.Linear(in_dim, n_nodes))
    if act.startswith("ReLU"):
        layers += [nn.ReLU(), nn.BatchNorm1d(n_nodes)]
    else:
        layers.append(nn.SELU())
    layers.append(nn.Dropout(0.2))
    in_dim = n_nodes
layers.append(nn.Linear(in_dim, 1, bias=False))
net = nn.Sequential(*layers)
n_params = sum(p.numel() for p in net.parameters())

fig, ax = plt.subplots(figsize=(10, 3.4))
xs = [0.6] + [1.9 + 2.1 * i for i in range(n_layers)] + [1.9 + 2.1 * n_layers]
sizes = [10] + [n_nodes] * n_layers + [1]
names = ["inputs\n(10 sensors)"] + [f"hidden {i + 1}\n{n_nodes} nodes"
                                    for i in range(n_layers)] + \
        ["output\nĥ(x) = log-risk"]
colours = ["#e3f2fd"] + ["#fff3e0"] * n_layers + ["#ffcdd2"]
for k, (x0, size, nm, col) in enumerate(zip(xs, sizes, names, colours)):
    shown = min(size, 6)
    for j in range(shown):
        yy = 1.6 + (j - (shown - 1) / 2) * 0.42
        ax.add_patch(plt.Circle((x0, yy), 0.15, facecolor=col,
                                edgecolor="#37474f", linewidth=1.2))
    if size > shown:
        ax.text(x0, 1.6 - (shown / 2) * 0.42 - 0.16, "⋮", ha="center",
                fontsize=11)
    ax.text(x0, 0.35, nm, ha="center", fontsize=8, color="#37474f")
    if k < len(xs) - 1:
        ax.annotate("", xy=(xs[k + 1] - 0.25, 1.6), xytext=(x0 + 0.25, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color="#90a4ae",
                                    linewidth=1.6))
        mid = (x0 + xs[k + 1]) / 2
        label = (act.split(" +")[0] + ("\n+BatchNorm" if act.startswith("ReLU")
                                       else "") + "\n+Dropout"
                 if k < len(xs) - 2 else "linear\n(no activation,\nno bias)")
        ax.text(mid, 2.35, label, ha="center", fontsize=7, color="#546e7a")
ax.set_xlim(0, xs[-1] + 1.4)
ax.set_ylim(0, 3.0)
ax.axis("off")
ax.set_title(f"DeepSurv — {n_params:,} parameters", fontsize=11)
st.pyplot(fig)
plt.close(fig)
st.caption(
    f"Compare: a linear Cox model on these 10 sensors has exactly **10** "
    f"parameters. This network has **{n_params:,}** — which is precisely why "
    "it can fit shapes Cox can't, and precisely why it needs dropout, L2 and "
    "a careful search not to overfit 1,000 machines."
)

st.header("2 · Coding the loss — the one part you must get right")
st.markdown(
    """
The subtle part is the risk set: for each failure $i$, the denominator sums
over **everyone still at risk**, including the censored. The standard
vectorized trick is to **sort by time descending**, so the risk set becomes a
running *cumulative sum from the top* — then `logcumsumexp` computes every
denominator in one pass.

Below, our from-scratch loss is checked against **pycox's official
implementation** (`CoxPHLoss`), which is the library DeepSurv/DeepHit ship in:
"""
)
show_example(
    '''import numpy as np
import torch
from pycox.models.loss import CoxPHLoss

# 6 machines: log-risks from the network, times, events
h = torch.tensor([0.5, -0.3, 1.2, 0.1, -0.8, 0.6])
T = torch.tensor([5.0, 2.0, 8.0, 3.0, 11.0, 4.0])  # note: no tied times
E = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])   # 0 = censored

# --- our from-scratch version: the formula, transcribed literally ---
def cox_nll(h, T, E):
    total = 0.0
    n_events = 0
    for i in range(len(T)):
        if E[i] == 1:                       # only OBSERVED failures contribute
            at_risk = T >= T[i]             # the risk set R(T_i)
            log_denom = torch.logsumexp(h[at_risk], dim=0)
            total = total + (h[i] - log_denom)
            n_events += 1
    return -total / n_events                # negative, averaged over events

mine = cox_nll(h, T, E)
print("our loss:  ", round(mine.item(), 6))

# --- pycox's official implementation ---
official = CoxPHLoss()(h.reshape(-1, 1), T, E)
print("pycox loss:", round(official.item(), 6))
print("agree:", torch.allclose(mine, official, atol=1e-5))''',
    """
- `if E[i] == 1` — **only observed failures contribute a term**. A censored machine never appears as an `i`… but it *does* appear in other machines' risk sets. That asymmetry is the whole of survival analysis in one line of code.
- `at_risk = T >= T[i]` — the risk set: everyone whose observed time is at least `T[i]`, i.e. still being watched when machine `i` failed.
- `torch.logsumexp(h[at_risk], dim=0)` — computes $\\log \\sum e^{h_j}$ **stably**. Never write `torch.log(torch.exp(h).sum())`: with a log-risk of 90, `exp(90)` overflows to infinity and your loss becomes `nan`. `logsumexp` internally subtracts the max first — the identical trick you used for softmax on the activations page.
- `-total / n_events` — negate (so we minimise) and average over the number of events. Averaging rather than summing keeps the loss scale independent of dataset size, which keeps a given learning rate sensible.
- The final comparison with pycox's `CoxPHLoss` is the verification that matters: **if these two numbers didn't match, everything downstream in this section would be worthless.**
""",
)

st.subheader("A subtlety a statistician will be asked about: tied event times")
st.markdown(
    r"""
Notice the example above deliberately has **no two machines failing at the
same time**. That was not laziness — it hides a real issue you should be able
to speak to.

When two machines fail at *exactly* the same recorded time, "the risk set at
$T_i$" becomes ambiguous: does the other tied machine belong in the
denominator or not? Classical survival analysis has named answers:

- **Breslow's approximation** — keep *all* tied machines in each other's risk
  sets (simple; what the literal `T >= T[i]` mask does). Biased when ties are
  heavy.
- **Efron's approximation** — a more careful adjustment; **lifelines'
  default**, and generally preferred.
- **Exact** methods — enumerate the orderings; correct but expensive.

The fast `logcumsumexp` implementation used by pycox (and shown below)
effectively **breaks ties by sort order**, which is a third thing again. The
practical consequences:

- On **tie-free data all these agree exactly** — you can verify this yourself
  in the sandbox, and the code snippets on this page do.
- On data with ties they differ *slightly*. With continuous times ties are
  rare and the difference is negligible. But if your survival times are
  recorded coarsely — **whole months, or whole overs** — ties can be
  everywhere, and the choice starts to matter.

This is worth a sentence in your dissertation's methods section, because it's
a discrepancy between "the Cox model as taught in statistics" and "the Cox
loss as implemented in deep-learning libraries", and few students notice it.
"""
)

with st.expander("🎓 Deeper statistics — two more things a viva could probe: "
                 "mini-batches, and what the L2 penalty *is*"):
    st.markdown(
        r"""
**1 · The partial likelihood is not a sum over observations — and SGD
quietly assumes it is.** Every loss you've minimised so far decomposes as
$\ell(\theta) = \sum_i \ell_i(\theta)$ with one independent term per data
point; that's what makes mini-batch gradients *unbiased* estimates of the
full gradient, which is the entire justification for SGD. The Cox loss
breaks this: each failure's term contains a **log-sum over its whole risk
set**, coupling every observation to every other. When deep-survival code
trains on mini-batches (as pycox does for large data), the risk set is
silently redefined as *"those still at risk **within this batch**"* — a
random subsample of the true denominator. The resulting gradient is **not**
an unbiased estimate of the full-data gradient (a log of a sample mean is a
biased estimate of the log of the population mean — Jensen's inequality).
In practice it works fine because the bias shrinks with batch size and the
ranking signal survives subsampling, but you should know it's an
approximation being made, not a theorem. On our small datasets we sidestep
it entirely by training full-batch — every risk set is exact.

**2 · Ridge is a Gaussian prior wearing a loss-function costume.** Adding
$\lambda\|\theta\|_2^2$ to a negative log-likelihood is *algebraically
identical* to putting an independent $N(0, \sigma^2)$ prior on every weight
(with $\lambda \propto 1/\sigma^2$) and finding the **MAP estimate** —
maximise $\log p(\text{data}\mid\theta) + \log p(\theta)$ and the second
term *is* $-\lambda\|\theta\|^2$ plus a constant. So DeepSurv's loss is a
penalised partial likelihood, and a Bayesian would read the whole objective
as "posterior mode under a Gaussian prior on the network's weights". This
also explains *why* weight decay fights overfitting in likelihood language:
it shrinks the effective parameter count, exactly as ridge regression
shrinks coefficients — the same mathematics you met on the regularization
page, now applied to $\theta$ instead of $\beta$.
"""
    )

st.header("3 · Watch the loss respond to the network's opinions")
st.markdown(
    "Six machines. You control the log-risk the network assigns to **M1** "
    "(which failed early, at month 2) and to **M5** (which survived longest, "
    "to month 11). The loss is recomputed live by our from-scratch function."
)
c1, c2 = st.columns(2)
h1 = c1.slider("network's log-risk for M1 (failed EARLY at month 2)",
               -3.0, 3.0, 0.0, 0.1)
h5 = c2.slider("network's log-risk for M5 (survived LONGEST, month 11)",
               -3.0, 3.0, 0.0, 0.1)

h_vec = torch.tensor([h1, 0.2, -0.4, 0.3, h5, 0.1], dtype=torch.float32)
T_vec = torch.tensor([2.0, 4.0, 5.0, 7.0, 11.0, 9.0])
E_vec = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])


def cox_nll(h, T, E):
    total, n_ev = 0.0, 0
    for i in range(len(T)):
        if E[i] == 1:
            at_risk = T >= T[i]
            total = total + (h[i] - torch.logsumexp(h[at_risk], dim=0))
            n_ev += 1
    return -total / n_ev


loss_val = float(cox_nll(h_vec, T_vec, E_vec))

grid = np.linspace(-3, 3, 60)
losses_m1 = [float(cox_nll(torch.tensor([g, 0.2, -0.4, 0.3, h5, 0.1]),
                           T_vec, E_vec)) for g in grid]
losses_m5 = [float(cox_nll(torch.tensor([h1, 0.2, -0.4, 0.3, g, 0.1]),
                           T_vec, E_vec)) for g in grid]

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(grid, losses_m1, color="#c62828", linewidth=2.2,
        label="varying M1's log-risk (failed EARLY)")
ax.plot(grid, losses_m5, color="#1565c0", linewidth=2.2,
        label="varying M5's log-risk (survived LONGEST)")
ax.scatter([h1], [loss_val], s=110, color="#c62828", zorder=5)
ax.scatter([h5], [loss_val], s=110, color="#1565c0", zorder=5)
ax.set_xlabel("log-risk assigned by the network")
ax.set_ylabel("Cox negative log partial likelihood")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.metric("current loss", f"{loss_val:.4f}")
st.success(
    """
**Read the two curves — this is the model's entire value system.**

The red curve (M1, which failed early) slopes **down** as you raise its
log-risk: the loss *rewards* the network for calling an early-failing machine
high-risk. The blue curve (M5, the long survivor) slopes **up**: the loss
*punishes* the network for calling a survivor high-risk.

So the loss never asks the network to predict *when* a machine fails. It only
pushes it to **rank** machines correctly — high risk for those who fail
sooner. That is why the C-index (a pure ranking metric) is the natural way to
score this model, and why DeepSurv outputs a **risk score with no units**
rather than a survival time.
"""
)

st.header("4 · Everything together: a real DeepSurv fit, from scratch")
show_example(
    experiments.SNIPPETS["d4_example"],
    """
- `machines(risk="nonlinear")` — our data generator, which reproduces the paper's gaussian risk function $h(x) = \\log(\\lambda_{max})\\exp(-(x_0^2+x_1^2)/2r^2)$. Only 2 of the 10 sensors matter, and they matter in a shape no straight line can capture.
- `mu, sd = X.mean(0), X.std(0)` and standardizing — this is one of the paper's stated training techniques, and it's computed on **training data only** (the leakage rule from Section 2).
- `torch.argsort(T, descending=True)` + `torch.logcumsumexp` — the vectorized risk-set trick. After sorting longest-time-first, position *i*'s risk set is exactly *everyone from 0 to i*, so a cumulative logsumexp gives all denominators at once. This is what makes the loss fast enough to train on.
- `* E` then `/ E.sum()` — multiplying by the event indicator zeroes out the censored machines' terms (they contribute no numerator), and we average over the number of true events.
- `weight_decay=1e-4` — **this is the $\\lambda\\|\\theta\\|^2_2$ term**. In PyTorch, L2 regularization is applied by the optimiser, not written into the loss. Same maths, different plumbing.
- `concordance_index(Tte, -risk, Ete)` — the minus sign again: lifelines wants higher = survives longer, our network outputs higher = riskier.
""",
    key="d4_example",
    heavy=True,
    est="~10 s",
)

guided_sandbox(
    key="d4",
    heavy=True,
    est="~25 s",
    steps="""
1. **Step 1** — write `cox_nll(h, T, E)` the **slow, literal way**: loop over
   failures, build the risk-set mask `T >= T[i]`, and accumulate
   `h[i] - torch.logsumexp(h[at_risk], dim=0)`. Return the negative mean over
   events. Print it for the little 6-machine example.
2. **Step 2** — verify against pycox: `CoxPHLoss()(h.reshape(-1, 1), T, E)`.
   They must agree. **Do not proceed until they do** — every result in this
   section depends on this function being right.
3. **Step 3** — now write the **fast** version using
   `torch.argsort(T, descending=True)` and `torch.logcumsumexp`, and check it
   agrees with your slow version. Then **create a tie on purpose** (set
   `T[3] = 2.0`, matching `T[1]`) and re-compare: the two versions now
   *disagree*. You've just reproduced the Breslow-vs-sort-order tie problem
   discussed above — worth understanding, not fearing.
4. **Step 4 (stretch)** — build the DeepSurv network, train it for 300 epochs
   on the nonlinear machines with `Adam(lr=0.01, weight_decay=1e-4)`, and
   print the test C-index. Then set `weight_decay=0` and compare — does the
   L2 penalty help on this dataset?
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
from utils.mockdata import machines

# the little example, for checking the loss
h = torch.tensor([0.5, -0.3, 1.2, 0.1, -0.8, 0.6])
T = torch.tensor([5.0, 2.0, 8.0, 2.0, 11.0, 4.0])
E = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

# the real data, for step 4
X, Ttr, Etr = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)
mu, sd = X.mean(0), X.std(0)
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Xts = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
print("ready:", Xs.shape, "training machines")

# Step 1: the slow, literal cox_nll(h, T, E)


# Step 2: check it against pycox's CoxPHLoss


# Step 3: the fast logcumsumexp version; check it agrees


# Step 4 (stretch): train DeepSurv, print test C-index, with/without L2
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
from utils.mockdata import machines

h = torch.tensor([0.5, -0.3, 1.2, 0.1, -0.8, 0.6])
T = torch.tensor([5.0, 2.0, 8.0, 2.0, 11.0, 4.0])
E = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

def cox_nll_slow(h, T, E):
    total, n_ev = 0.0, 0
    for i in range(len(T)):
        if E[i] == 1:
            at_risk = T >= T[i]
            total = total + (h[i] - torch.logsumexp(h[at_risk], dim=0))
            n_ev += 1
    return -total / n_ev

slow = cox_nll_slow(h, T, E)
official = CoxPHLoss()(h.reshape(-1, 1), T, E)
print(f"slow version: {slow.item():.6f}")
print(f"pycox:        {official.item():.6f}")
print("agree:", torch.allclose(slow, official, atol=1e-5))

def cox_nll(h, T, E):
    order = torch.argsort(T, descending=True)
    h, E = h[order], E[order]
    log_denom = torch.logcumsumexp(h, dim=0)
    return -((h - log_denom) * E).sum() / E.sum()

fast = cox_nll(h, T, E)
print(f"fast version: {fast.item():.6f}  agrees:",
      torch.allclose(slow, fast, atol=1e-5))

T_tied = T.clone()
T_tied[3] = 2.0          # now machines 1 and 3 fail at the SAME time
print(f"\\nwith a tie -> slow (Breslow-style): "
      f"{cox_nll_slow(h, T_tied, E).item():.6f}")
print(f"              fast (sort-order):   "
      f"{cox_nll(h, T_tied, E).item():.6f}  <- they diverge")

X, Ttr, Etr = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)
mu, sd = X.mean(0), X.std(0)
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Xts = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
Tt = torch.tensor(Ttr, dtype=torch.float32)
Et = torch.tensor(Etr, dtype=torch.float32)

for wd in [1e-4, 0.0]:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
        nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
        nn.Linear(32, 1, bias=False))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=wd)
    for epoch in range(300):
        net.train()
        opt.zero_grad()
        loss = cox_nll(net(Xs).squeeze(1), Tt, Et)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        risk = net(Xts).squeeze(1).numpy()
    c = concordance_index(Tte, -risk, Ete)
    print(f"weight_decay={wd:g}: test C-index {c:.4f}")''',
)
