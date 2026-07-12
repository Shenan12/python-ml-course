import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.sandbox import guided_sandbox, show_example

st.title("🕰️ Faraggi–Simon: Why Earlier Neural Networks Failed")
st.markdown(
    """
Replacing $\\beta^\\top x$ with a neural network is not a new idea. **Faraggi
and Simon proposed it in 1995** — a single hidden layer feeding one output
node, trained on the Cox partial likelihood. Structurally, that *is* DeepSurv.

And it didn't work. For roughly two decades, neural survival models
**failed to beat the plain linear Cox model** in study after study, and the
approach was largely abandoned.

So the honest question a supervisor will ask you in a viva: *"If the idea is
from 1995, what exactly did Katzman et al. contribute in 2018?"* This page is
your answer — and it's an unusually clean case study in how deep learning
progressed, because **the architecture barely changed; the training did.**
"""
)

st.warning(
    "**A note on sourcing.** The DeepSurv paper's own account is brief: it "
    "describes the Faraggi–Simon network as a neural extension of the Cox "
    "model that historically failed to demonstrate improvement over the "
    "linear Cox model, and attributes DeepSurv's success in part to modern "
    "training techniques not available earlier. The itemised diagnosis below "
    "is my reconstruction from the broader deep-learning literature of that "
    "period, not a claim-by-claim quotation from Katzman et al. Treat the "
    "table as well-founded context, not as citable paper content — and "
    "verify anything you plan to put in your dissertation against the paper "
    "text directly."
)

st.header("1 · The diagnosis: everything around the network was missing")
st.markdown(
    """
| What a 1995 net had | What DeepSurv (2018) has | Why it decides the outcome |
|---|---|---|
| **sigmoid / tanh** activations | **ReLU or SELU** | Sigmoid's derivative is ≤ 0.25 and ≈ 0 once saturated, so blame vanishes as it flows back. You *measured* this on the activations page: after 10 layers, best case, gradient is multiplied by 0.25¹⁰ ≈ 0.000001. Deep nets simply couldn't be trained. |
| **plain SGD**, hand-tuned fixed learning rate | **Adam**, momentum, learning-rate decay | Getting a good result depended on guessing a learning rate that worked. Adam adapts per-parameter step sizes automatically. |
| **no dropout** (invented 2012), **no weight-decay culture** | dropout (searched 0.11–0.66) + large L2 | Small survival datasets (WHAS: 1,638 units) and a flexible model = overfitting. Without modern regularizers, a neural net memorises and generalises worse than a rigid linear model. (Batch norm also existed by 2018, though the DeepSurv paper itself doesn't use it.) |
| **little / no hyperparameter search** | **random search** over depth, width, learning rate, dropout, L2 | A neural net has many knobs and is *brutally* sensitive to them. A badly-configured net loses to Cox. A well-searched one wins. This may be the single biggest factor. |
| **no standardization** emphasised | inputs standardized | You saw on the gradient-descent page how unscaled inputs stretch the loss surface into a ravine that descent crawls along. |
| **tiny compute** | GPUs, seconds per fit | You can't random-search 100 configurations if each fit takes an afternoon. Search *is* compute. |

The lesson generalises far beyond survival analysis, and it's worth
internalising for your career: **a model class is not good or bad in the
abstract — it is good or bad *as trained*.** The same architecture, given
1995's toolkit, loses to a linear model; given 2018's, it beats it. When you
read "method X doesn't work," always ask *how hard did they try?*
"""
)

st.header("2 · See it happen: the same network, two eras of training")
st.markdown(
    "Both networks below have **identical architecture** (one hidden layer, "
    "16 units) and see identical data. Only the *training* differs. This is "
    "an illustration of the mechanism using an ordinary regression task — "
    "not a reproduction of any published survival experiment — but the "
    "failure mode is exactly the one the table describes."
)

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (200, 1))
y_true = np.sin(X[:, 0]) * 2 + 0.3 * X[:, 0] ** 2
y = y_true + rng.normal(0, 0.3, 200)


def train_mlp(activation, lr, optimiser, epochs=400, seed=0):
    """A tiny MLP trained from scratch, so both 'eras' are honestly compared."""
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.5, (1, 16))
    b1 = np.zeros(16)
    W2 = r.normal(0, 0.5, (16, 1))
    b2 = np.zeros(1)
    m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]   # Adam state
    v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    hist = []
    for ep in range(1, epochs + 1):
        z1 = X @ W1 + b1
        if activation == "sigmoid":
            a1 = 1 / (1 + np.exp(-z1))
            da = a1 * (1 - a1)
        else:                                            # relu
            a1 = np.maximum(0, z1)
            da = (z1 > 0).astype(float)
        pred = (a1 @ W2 + b2).ravel()
        err = pred - y
        hist.append(float(np.mean(err ** 2)))
        gpred = (2 * err / len(y))[:, None]
        gW2 = a1.T @ gpred
        gb2 = gpred.sum(axis=0)
        gh = gpred @ W2.T * da
        gW1 = X.T @ gh
        gb1 = gh.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        if optimiser == "Adam":
            for k, (p, g) in enumerate(zip((W1, b1, W2, b2), grads)):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g ** 2
                mhat = m[k] / (1 - 0.9 ** ep)
                vhat = v[k] / (1 - 0.999 ** ep)
                p -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        else:                                            # plain SGD
            for p, g in zip((W1, b1, W2, b2), grads):
                p -= lr * g
    return hist, (W1, b1, W2, b2, activation)


def predict(params, xs):
    W1, b1, W2, b2, act = params
    z1 = xs @ W1 + b1
    a1 = (1 / (1 + np.exp(-z1)) if act == "sigmoid" else np.maximum(0, z1))
    return (a1 @ W2 + b2).ravel()


hist_old, p_old = train_mlp("sigmoid", 0.01, "SGD")
hist_new, p_new = train_mlp("relu", 0.01, "Adam")

xs = np.linspace(-3, 3, 200)[:, None]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(hist_old, color="#c62828", linewidth=2,
             label=f"1995 kit: sigmoid + plain SGD (final MSE "
                   f"{hist_old[-1]:.3f})")
axes[0].plot(hist_new, color="#2e7d32", linewidth=2,
             label=f"2018 kit: ReLU + Adam (final MSE {hist_new[-1]:.3f})")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("training MSE")
axes[0].set_yscale("log")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_title("same network, same data, same learning rate", fontsize=10)

axes[1].scatter(X[:, 0], y, s=10, color="#b0bec5", label="data")
axes[1].plot(xs.ravel(), np.sin(xs.ravel()) * 2 + 0.3 * xs.ravel() ** 2,
             color="#37474f", linestyle="--", linewidth=1.5,
             label="true pattern")
axes[1].plot(xs.ravel(), predict(p_old, xs), color="#c62828", linewidth=2,
             label="1995 kit")
axes[1].plot(xs.ravel(), predict(p_new, xs), color="#2e7d32", linewidth=2,
             label="2018 kit")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].set_title("what each one actually learned", fontsize=10)
st.pyplot(fig)
plt.close(fig)

st.error(
    f"**The 1995-kit network is still floundering** (MSE "
    f"{hist_old[-1]:.3f}) while the identical network with modern training "
    f"has essentially solved the problem (MSE {hist_new[-1]:.3f}) — "
    f"**{hist_old[-1] / hist_new[-1]:.0f}× better**, from nothing but the "
    "activation function and the optimiser. Now imagine that handicapped "
    "network being compared to a well-understood linear Cox model on a "
    "1,638-unit dataset, and you can see exactly how the field concluded, "
    "for twenty years, that neural survival models 'don't work'."
)

st.header("3 · So what is DeepSurv's actual contribution?")
st.markdown(
    """
State it precisely, because this is the sentence your dissertation's
literature review needs:

> DeepSurv is a **deep** (rather than single-hidden-layer) feed-forward
> network trained on the Cox negative log partial likelihood, whose
> contribution is not architectural novelty but the demonstration that —
> **with modern activations, optimisers, regularization and hyperparameter
> search** — such a network *does* outperform the linear Cox model on real
> clinical data, and can be used to build a personalised treatment
> recommender.

Note what is honest about that framing: **DeepSurv beats Cox precisely when
the true risk is nonlinear.** When the truth *is* linear, the paper's own
simulation shows Cox and DeepSurv performing essentially identically (0.779
vs 0.778) — as they should, since Cox is then the correctly-specified model
and a flexible model can only match it. A more powerful model is not a
uniformly better one; it's an insurance policy against misspecification. You
will reproduce both halves of that result yourself on the simulation page.
"""
)

with st.expander("🎓 Deeper statistics — the same point in estimation-theory "
                 "language"):
    st.markdown(
        r"""
The Cox model with linear $\beta^\top x$ is **nested** inside DeepSurv (a
network with no hidden layers *is* $\beta^\top x$). So the comparison is a
classic parametric-vs-flexible trade, and the standard decomposition applies
to the risk function each model learns:

$$\text{error} \;=\; \underbrace{\text{approximation error}}_{\text{can the model class express the truth?}} \;+\; \underbrace{\text{estimation error}}_{\text{how well can you fit it from } n \text{ observations?}}$$

- **Truth linear** → Cox has zero approximation error *and* is the
  (semiparametrically) efficient estimator: its $O(1/\sqrt n)$ estimation
  error has the smallest possible constant. DeepSurv can at best match it,
  and pays extra estimation error for capacity it can't use. Hence the
  paper's 0.779 vs 0.778 — a tie is the *predicted* outcome, not a
  disappointment.
- **Truth nonlinear** → Cox's approximation error is a fixed bias that no
  amount of data removes ($\hat\beta$ converges, quickly and with beautiful
  confidence intervals, to the best *linear* approximation of a non-linear
  truth — precisely wrong). DeepSurv's approximation error is ~0 and its
  estimation error shrinks with $n$. Flexibility wins, *if* $n$ and the
  regularisation can control the variance.

That last clause is the Faraggi–Simon story in one line: in 1995 the
estimation-error term was effectively unbounded (no dropout, no weight
decay, optimisers that couldn't find the minimum), so the flexible model
lost even when the truth was nonlinear. The 2018 toolkit didn't change the
model class — it shrank the estimation error until the approximation-error
advantage could finally show.
"""
    )

show_example(
    '''# The Faraggi-Simon network and DeepSurv, side by side in code.
# Spot the architectural difference. (Hint: there barely is one.)
import torch.nn as nn

faraggi_simon_1995 = nn.Sequential(
    nn.Linear(10, 8),
    nn.Sigmoid(),                 # <- the era's activation
    nn.Linear(8, 1, bias=False),  # single output = the log-risk h(x)
)

deepsurv_2018 = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),                    # <- modern activation
    nn.BatchNorm1d(32),           # <- didn't exist until 2015
    nn.Dropout(0.2),              # <- didn't exist until 2012
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.2),
    nn.Linear(32, 1, bias=False), # same single output, same loss
)

print("Faraggi-Simon params:", sum(p.numel() for p in
                                   faraggi_simon_1995.parameters()))
print("DeepSurv params:     ", sum(p.numel() for p in
                                   deepsurv_2018.parameters()))
print()
print("Same loss function (Cox partial likelihood).")
print("Same single linear output node = the log-risk.")
print("The difference is depth, activations, regularization -")
print("and everything OUTSIDE the model: the optimiser and the search.")''',
    """
- Both networks end in `nn.Linear(..., 1, bias=False)` — **one output neuron, no activation on it**. That single number *is* $\\hat h_\\theta(x)$, the log-risk that slots into the Cox model in place of $\\beta^\\top x$.
- Why `bias=False`? Because a constant added to every machine's log-risk **cancels out of the partial likelihood entirely** (it appears in both numerator and denominator of every softmax term). The bias is unidentifiable — a genuinely elegant detail you can now see straight from the formula on the previous page. Note the same is true of the Cox model itself: it has no intercept, for exactly this reason, with the baseline hazard absorbing it.
- The layer stack is the only structural difference, and it's modest. The real gap is in code that isn't shown here: the optimiser, the schedule, and the hyperparameter search.
""",
)

guided_sandbox(
    key="d3",
    steps="""
1. **Step 1** — build both networks with `nn.Sequential`: a Faraggi–Simon
   style net (`Linear(10, 8) → Sigmoid → Linear(8, 1, bias=False)`) and a
   DeepSurv-style one (two hidden layers of 32 with ReLU, BatchNorm, Dropout,
   then `Linear(32, 1, bias=False)`). Print each one's parameter count.
2. **Step 2** — confirm the bias-cancellation claim numerically. Take any
   risk-score vector `h`, compute the Cox partial-likelihood term
   `h[i] - logsumexp(h[risk_set])` for one failure, then add a constant `c`
   to *every* element of `h` and recompute. The value should be **identical**.
   (Use `from scipy.special import logsumexp`.) That's why the output layer
   needs no bias.
3. **Step 3** — push a batch of 20 machines through both nets and print the
   output shapes. Confirm both produce one number per machine.
4. **Step 4 (stretch)** — check the vanishing-gradient story yourself: run a
   backward pass on each net from a dummy loss, and print the gradient
   magnitude reaching the **first** layer's weights
   (`net[0].weight.grad.abs().mean()`). Compare sigmoid vs ReLU.
""",
    setup_code='''import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

torch.manual_seed(0)
X = torch.randn(20, 10)          # 20 machines, 10 sensors
h = np.array([0.4, -1.2, 2.0, 0.1, -0.6])   # 5 machines' log-risks
risk_set = [0, 1, 2, 3, 4]
print("ready")

# Step 1: build faraggi_simon and deepsurv nets; print parameter counts


# Step 2: show that adding a constant to every h leaves the Cox term unchanged


# Step 3: forward a batch through both; print output shapes


# Step 4 (stretch): compare gradient magnitude at layer 1, sigmoid vs ReLU
''',
    solution_code='''import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

torch.manual_seed(0)
X = torch.randn(20, 10)
h = np.array([0.4, -1.2, 2.0, 0.1, -0.6])
risk_set = [0, 1, 2, 3, 4]

faraggi_simon = nn.Sequential(nn.Linear(10, 8), nn.Sigmoid(),
                              nn.Linear(8, 1, bias=False))
deepsurv = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
    nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
    nn.Linear(32, 1, bias=False))
print("Faraggi-Simon params:", sum(p.numel() for p in
                                   faraggi_simon.parameters()))
print("DeepSurv params:     ", sum(p.numel() for p in deepsurv.parameters()))

term = h[2] - logsumexp(h[risk_set])
term_shifted = (h[2] + 7.5) - logsumexp(h[risk_set] + 7.5)
print(f"\\nCox term:            {term:.6f}")
print(f"after adding c=7.5:  {term_shifted:.6f}")
print("identical ->", np.isclose(term, term_shifted),
      "  so the output bias is unidentifiable")

print("\\noutput shapes:", faraggi_simon(X).shape, deepsurv(X).shape)

print("\\ngradient reaching the FIRST layer:")
for name, net in [("sigmoid (1995)", faraggi_simon), ("ReLU (2018)", deepsurv)]:
    net.zero_grad()
    net(X).sum().backward()
    g = net[0].weight.grad.abs().mean().item()
    print(f"  {name:16}: {g:.6f}")''',
)
