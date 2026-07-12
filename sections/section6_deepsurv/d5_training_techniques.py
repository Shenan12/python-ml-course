import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section6_deepsurv import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🛠️ DeepSurv's Training Techniques")
st.markdown(
    """
The previous page argued that DeepSurv's contribution is *how it's trained*.
This page takes each technique the paper names, explains what it does, and
**demonstrates it working on our machine data**.

The paper names its techniques in one sentence — *"standardizing the input,
Scaled Exponential Linear Units (SELU) as the activation function, Adaptive
Moment Estimation (Adam) for the gradient descent algorithm, Nesterov
momentum, and learning rate scheduling"* — with the searched values in its
hyperparameter table (Table 3):

| Technique | What the paper reports |
|---|---|
| **Standardized inputs** | covariates normalized before training |
| **Activations** | SELU or ReLU (the search picked SELU for 5 of 7 experiments) |
| **Dropout** | tuned per dataset: 0.11–0.66 across the experiments |
| **Optimiser** | SGD (both simulations) or **Adam** (all real datasets), with **Nesterov momentum** (0.84–0.94) |
| **Learning-rate decay** | inverse-time decay: lr_t = lr₀ / (1 + t · decay), rates 0.0003–0.006 |
| **L2 / weight decay** | the λ‖θ‖² term — searched values 2.0–16.1 |

Note the L2 coefficients are *large* by deep-learning standards (into the
double digits) — a sign of how hard these small, censored datasets push back
against a flexible model.

*(This page also covers **gradient clipping** below. Honesty note: clipping
is **not** named in the DeepSurv paper — it's standard craft for any loss
with an exponential inside, and you'll meet it constantly in practice, so we
teach it here rather than pretend the topic doesn't exist. It is our
addition, clearly flagged.)*
"""
)

st.header("1 · Standardizing the inputs")
st.markdown(
    r"""
The first and cheapest win. Recall the Section 2 gradient-descent page: when
features live on wildly different scales, the loss surface becomes a stretched
ravine and descent zig-zags down it slowly. Standardizing —
$x \leftarrow (x - \mu)/\sigma$, with $\mu, \sigma$ from the **training set
only** — makes the bowl round.

For survival data there's a second reason, specific to this loss. The log-risk
$\hat h_\theta(x)$ gets **exponentiated** inside the partial likelihood. If
unscaled inputs let the network produce a log-risk of 90, then $e^{90}$
overflows and your loss becomes `nan`. (Using `logsumexp` protects the
*summation*, but nothing protects you from weights that have already blown up.)

To find out whether this actually matters, we train the same network **four
times**: raw vs standardized sensors × plain SGD vs Adam. The sensors are given
deliberately awful scales (one spans thousands, another hundredths), as real
sensor data does. The results below are real, and — fair warning — they did not
say what I expected them to.
"""
)


@st.cache_data(show_spinner="Training all four configurations for real…")
def compare_standardization_live():
    return experiments.standardization_experiment()


std_res = artifacts.load("d5_standardization")
if st.session_state.get("_d5_live"):
    std_res = compare_standardization_live()
    d5_note = "just trained live on your machine"
elif std_res is not None:
    d5_note = artifacts.provenance(std_res)
else:
    std_res = compare_standardization_live()
    d5_note = "computed live"

cap, btn = st.columns([3, 1])
cap.caption(f"📦 Four real training runs — {d5_note}.")
if btn.button("▶ Re-train live (~5 s)"):
    st.session_state["_d5_live"] = True
    st.rerun()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, opt in zip(axes, ["SGD", "Adam"]):
    for scaling, colour in [("raw", "#c62828"), ("standardized", "#2e7d32")]:
        key = f"{scaling}_{opt}"
        hist = np.asarray(std_res[f"{key}_hist"], dtype=float)
        diverged = bool(std_res[f"{key}_diverged"])
        c = float(std_res[f"{key}_cindex"])
        label = (f"{scaling} — DIVERGED (NaN)" if diverged
                 else f"{scaling} — C-index {c:.3f}")
        ax.plot(np.where(np.isfinite(hist), hist, np.nan), color=colour,
                linewidth=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Cox negative log partial likelihood")
    ax.set_title(f"optimiser: {opt}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

rows = []
for scaling in ["raw", "standardized"]:
    row = {"input scaling": scaling}
    for opt in ["SGD", "Adam"]:
        k = f"{scaling}_{opt}"
        row[opt] = ("💥 diverged (NaN)" if std_res[f"{k}_diverged"]
                    else f"C-index {float(std_res[f'{k}_cindex']):.3f}")
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.warning(
    """
**The result is more interesting than the textbook slogan — so let's report
what actually happened rather than what I expected.**

- **Plain SGD on raw sensors: the network diverges to `NaN` outright.** The
  huge-scale sensors produce huge gradients, the weights explode, the log-risk
  overflows inside `exp(·)`, and training dies. Total failure.
- **Plain SGD on standardized sensors: it trains.** Not brilliantly (SGD is
  slow here), but it *runs*.
- **Adam, however, barely cares.** Raw and standardized give nearly the same
  C-index. Why? Adam divides each parameter's step by a running estimate of
  its own gradient magnitude — so a feature with gigantic gradients gets
  proportionally tiny steps. **Adam is itself a partial substitute for
  standardizing.**

So the honest lesson is *not* "standardize or your C-index collapses". It is:
**standardization is insurance against catastrophic divergence.** With a modern
adaptive optimiser you may get away without it; with plain SGD you will not. It
costs one line, it removes an entire class of failure, and it makes the loss
surface rounder (the Section 2 ravine) — so the paper does it, and so should
you. But now you know *why*, and you know what's actually doing the work.
"""
)

st.header("2 · Learning-rate decay")
st.markdown(
    r"""
The paper applies **inverse-time decay**:

$$\text{lr}_t = \frac{\text{lr}_0}{1 + t \cdot \text{decay}}$$

The logic is the one you felt on the gradient-descent page: **early on you
want big steps** (you're far from the minimum, and speed matters), **later you
want small ones** (you're near the bottom, and big steps just bounce you
around the valley without settling). A decaying schedule gives you both.
"""
)
c1, c2 = st.columns(2)
lr0 = c1.select_slider("initial learning rate lr₀", [0.001, 0.01, 0.1],
                       value=0.01)
decay = c2.select_slider("decay rate", [0.0, 0.0001, 0.001, 0.005, 0.05],
                         value=0.001)
epochs_ax = np.arange(0, 500)
lrs = lr0 / (1 + epochs_ax * decay)
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(epochs_ax, lrs, color="#6a1b9a", linewidth=2.2)
ax.set_xlabel("epoch t")
ax.set_ylabel("learning rate")
ax.grid(alpha=0.25)
ax.set_title(f"lr₀={lr0} , decay={decay} → after 500 epochs the step size is "
             f"{lrs[-1]:.5f} ({lrs[-1] / lr0:.0%} of the original)",
             fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
if decay == 0.0:
    st.info("Decay = 0 means a **constant** learning rate — the schedule is "
            "switched off. Perfectly valid (Adam already adapts step sizes), "
            "and it's what most of our earlier pages did.")
st.caption(
    "The paper's Table 3 reports decay rates between 0.0003 and 0.006 across "
    "its experiments — note how gentle that is: even at 0.005, after 500 "
    "epochs the rate has only fallen to about a third of its initial value."
)

st.header("3 · Momentum, Nesterov, and Adam")
st.markdown(
    r"""
The paper uses momentum (Nesterov, ≈0.84–0.91) and Adam. Here's the intuition
chain, building on Section 2:

- **Plain SGD**: step directly downhill from where you stand. In a stretched
  ravine, it zig-zags across the walls and crawls along the floor.
- **Momentum**: keep a running *velocity* — a decayed average of past
  gradients — and step along that. Zig-zag components cancel out (they point
  opposite ways on alternate steps), while the consistent downhill component
  accumulates. Physically: a **heavy ball rolling** rather than a cautious
  hiker. Momentum 0.9 means "90% of last step's velocity carries over".
- **Nesterov momentum**: a refinement that computes the gradient at the point
  where the momentum is *about to take you*, rather than where you are — a
  "look-ahead" that lets it brake earlier before overshooting.
- **Adam**: momentum *plus* a per-parameter adaptive step size (it divides by
  a running estimate of each gradient's magnitude), so rarely-active
  parameters still get meaningful updates.

Watch all four descend the same stretched ravine — the classic picture, run
live:
"""
)


def descend(method, steps=60, lr=0.08, mom=0.9):
    """Descend the stretched bowl f(w) = 0.5*(w0^2/8 + 10*w1^2)."""

    def grad(w):
        return np.array([w[0] / 8, 10 * w[1]])

    w = np.array([-4.5, 1.6])
    v = np.zeros(2)
    m, s = np.zeros(2), np.zeros(2)
    path = [w.copy()]
    for t in range(1, steps + 1):
        if method == "SGD":
            w = w - lr * grad(w)
        elif method == "Momentum":
            v = mom * v + grad(w)
            w = w - lr * v
        elif method == "Nesterov":
            lookahead = w - lr * mom * v
            v = mom * v + grad(lookahead)
            w = w - lr * v
        else:                                        # Adam
            g = grad(w)
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g ** 2
            mhat, shat = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
            w = w - 0.35 * mhat / (np.sqrt(shat) + 1e-8)
        path.append(w.copy())
    return np.array(path)


W0, W1 = np.meshgrid(np.linspace(-5, 5, 120), np.linspace(-2, 2, 120))
Z = 0.5 * (W0 ** 2 / 8 + 10 * W1 ** 2)
fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.contour(W0, W1, Z, levels=22, cmap="Greys", alpha=0.55, linewidths=0.7)
for method, colour in [("SGD", "#c62828"), ("Momentum", "#ef6c00"),
                       ("Nesterov", "#1565c0"), ("Adam", "#2e7d32")]:
    p = descend(method)
    final = 0.5 * (p[-1, 0] ** 2 / 8 + 10 * p[-1, 1] ** 2)
    ax.plot(p[:, 0], p[:, 1], marker="o", markersize=2.5, linewidth=1.5,
            color=colour, label=f"{method} (final loss {final:.4f})")
ax.scatter([0], [0], marker="*", s=260, color="#ffd600", edgecolor="black",
           zorder=6, label="minimum")
ax.set_xlabel("w₀ (a gently-curved direction)")
ax.set_ylabel("w₁ (a steeply-curved direction)")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("60 steps each, same start, on a stretched loss surface",
             fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    "Plain SGD (red) bounces between the steep walls while barely progressing "
    "along the shallow floor — the exact pathology standardizing your inputs "
    "reduces, and momentum cures. Momentum and Nesterov damp the oscillation "
    "and pick up speed along the valley; Adam adapts its step per-direction "
    "and drives straight in. **This is why the optimiser mattered so much in "
    "1995 vs 2018**, and why it's worth naming precisely in your methods "
    "section."
)

with st.expander("🎓 Deeper statistics — you have used a better optimiser "
                 "than any of these, in every stats course"):
    st.markdown(
        r"""
When R fits a logistic regression or a Cox model, it doesn't use gradient
descent at all — it uses **Newton–Raphson / Fisher scoring**:

$$\theta_{t+1} = \theta_t + \mathcal{I}(\theta_t)^{-1}\, s(\theta_t)$$

step = *inverse information matrix* times the score. Multiplying by
$\mathcal{I}^{-1}$ rescales every direction by its curvature — it turns the
stretched ravine into a perfect bowl and typically converges in under ten
iterations, with the bonus that $\mathcal{I}(\hat\theta)^{-1}$ *is* the
coefficient covariance matrix (your standard errors) at the end.

So why doesn't deep learning use it? **Dimension.** With $p$ parameters the
information matrix is $p \times p$: for a Cox model with 10 covariates
that's a 10×10 solve (trivial); for even our small DeepSurv nets it's
~3,000×3,000, recomputed every step, and for real networks it's billions
squared — impossible. The whole modern-optimiser zoo is best read as
**cheap approximations to the Newton step**:

| optimiser | what it approximates |
|---|---|
| plain SGD | $\mathcal{I}^{-1} \approx$ (a constant) — no curvature at all |
| momentum / Nesterov | averages out the ravine's oscillation *as if* curvature were flattened |
| **Adam** | $\mathcal{I}^{-1} \approx$ a **diagonal** matrix estimated from running gradient magnitudes — per-parameter curvature, ignoring all cross-terms |

That reading also demystifies two facts from this page: Adam substituting
for standardization (a diagonal rescale is exactly what standardizing does
to the first layer), and lifelines agreeing with our gradient ascent on d2
while using far fewer iterations — it's running the real Newton step on a
problem small enough to afford it.
"""
    )

st.header("4 · Gradient clipping (our addition — not in the paper)")
st.markdown(
    r"""
The last safety net — and, as flagged at the top of the page, the one
technique here that Katzman et al. do **not** name. We include it because it
belongs to the same family of protections and you will see it in nearly every
real training script. Occasionally a batch produces an enormous gradient (a
near-tie in the risk set, an extreme covariate), and a single huge step throws
the network into a bad region it never recovers from — training "explodes",
and with an exponential inside your loss this is a real risk.

**Gradient clipping** rescales the gradient whenever its norm exceeds a
threshold $c$:

$$g \leftarrow g \cdot \frac{c}{\|g\|} \quad\text{if } \|g\| > c$$

Note it preserves the *direction* and only caps the *length* — you still step
downhill, just never absurdly far. One line in PyTorch:
`torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)`.

You already saw why this matters: the `raw + SGD` cell in the table above blew
up to `NaN`. Gradient clipping is the belt to standardization's braces — it
caps the damage a single monstrous gradient can do, whatever caused it.
"""
)

st.header("5 · The full recipe, assembled")
show_example(
    experiments.SNIPPETS["d5_example"],
    """
- `nn.SELU()` + `nn.AlphaDropout(0.1)` — the SELU variant the paper mentions. SELU is *self-normalising*: it keeps activations near zero-mean/unit-variance automatically, so it needs no BatchNorm. Its partner is `AlphaDropout`, which is dropout that preserves that self-normalising property (ordinary `Dropout` would break it). Use them as a pair — this is a detail people get wrong.
- `weight_decay=1e-4` — the λ‖θ‖² term, applied by the optimiser.
- The `for g in opt.param_groups: g["lr"] = ...` loop — inverse-time decay, implemented exactly as the paper's formula. (PyTorch also has `torch.optim.lr_scheduler.LambdaLR` to do this more tidily.)
- `clip_grad_norm_(net.parameters(), 1.0)` — **placed after `backward()` and before `step()`**, which is the only correct spot: the gradients must exist, and must be capped before they're applied. The trailing underscore means it modifies the gradients in place. (Clipping is our addition — the paper doesn't name it.)
- Each numbered comment is one of the techniques from the table above — the whole recipe in 25 lines.
""",
    key="d5_example",
    heavy=True,
    est="~10 s",
)

guided_sandbox(
    key="d5",
    heavy=True,
    est="~25 s",
    steps="""
1. **Step 1** — write `cox_nll(h, T, E)` using the fast `logcumsumexp`
   formulation (you wrote this on the previous page — from memory this time).
2. **Step 2** — build a DeepSurv net with **SELU + AlphaDropout** (2 hidden
   layers of 32) and train it for 300 epochs with `Adam(lr=0.01,
   weight_decay=1e-4)`. Print the test C-index.
3. **Step 3** — add the paper's two remaining techniques: inverse-time
   learning-rate decay (`lr = LR0 / (1 + epoch * 0.001)`, set inside
   `opt.param_groups`) and gradient clipping (`clip_grad_norm_` between
   `backward()` and `step()`). Does the C-index change?
4. **Step 4 (stretch)** — the ablation that makes the point: train **without
   standardizing** `X` (the setup gives you `X_badscale`, with realistic
   mixed sensor scales) and compare. Which single technique matters most on
   this data?
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from utils.mockdata import machines

X, T, E = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)

scales = np.array([1, 500, 0.01, 50, 1, 200, 0.05, 1, 10, 1000.0])
X_badscale, Xte_badscale = X * scales, Xte * scales   # for step 4

mu, sd = X.mean(0), X.std(0)
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Xts = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
Tt = torch.tensor(T, dtype=torch.float32)
Et = torch.tensor(E, dtype=torch.float32)
print("ready:", Xs.shape)

# Step 1: cox_nll(h, T, E) with logcumsumexp


# Step 2: SELU + AlphaDropout net, Adam + weight_decay, 300 epochs, C-index


# Step 3: add LR decay and gradient clipping


# Step 4 (stretch): the no-standardization ablation
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from utils.mockdata import machines

X, T, E = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)
scales = np.array([1, 500, 0.01, 50, 1, 200, 0.05, 1, 10, 1000.0])
X_badscale, Xte_badscale = X * scales, Xte * scales
Tt = torch.tensor(T, dtype=torch.float32)
Et = torch.tensor(E, dtype=torch.float32)

def cox_nll(h, T, E):
    order = torch.argsort(T, descending=True)
    h, E = h[order], E[order]
    return -((h - torch.logcumsumexp(h, 0)) * E).sum() / E.sum()

def run(Xtrain, Xtest, decay=0.0, clip=None):
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(10, 32), nn.SELU(), nn.AlphaDropout(0.1),
        nn.Linear(32, 32), nn.SELU(), nn.AlphaDropout(0.1),
        nn.Linear(32, 1, bias=False))
    LR0 = 0.01
    opt = torch.optim.Adam(net.parameters(), lr=LR0, weight_decay=1e-4)
    Xa = torch.tensor(Xtrain, dtype=torch.float32)
    Xb = torch.tensor(Xtest, dtype=torch.float32)
    for epoch in range(300):
        if decay:
            for g in opt.param_groups:
                g["lr"] = LR0 / (1 + epoch * decay)
        net.train()
        opt.zero_grad()
        loss = cox_nll(net(Xa).squeeze(1), Tt, Et)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
    net.eval()
    with torch.no_grad():
        risk = net(Xb).squeeze(1).numpy()
    return concordance_index(Tte, -risk, Ete)

mu, sd = X.mean(0), X.std(0)
Xs, Xts = (X - mu) / sd, (Xte - mu) / sd
print(f"standardized, plain Adam:            {run(Xs, Xts):.4f}")
print(f"+ LR decay + gradient clipping:      "
      f"{run(Xs, Xts, decay=0.001, clip=1.0):.4f}")

mu2, sd2 = X_badscale.mean(0), X_badscale.std(0)
print(f"NOT standardized (raw sensor scales): "
      f"{run(X_badscale, Xte_badscale):.4f}  <- the big one")''',
)
