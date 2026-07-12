import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from sections.section6_deepsurv import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🔀 The Treatment Recommender")
st.markdown(
    r"""
This is the part of the paper its *title* is about, and the part with the
most practical bite. A risk score is useful; a **recommendation** is more
useful. So: given a machine (or patient), which of two available treatments
should it get?

Katzman et al. define the **recommender function** as the difference in
predicted log-risk under the two treatments:

$$\text{rec}_{ij}(x) = \hat h_\theta(x, i) - \hat h_\theta(x, j)$$

i.e. *"what is the log-hazard ratio for this individual if we give it
treatment $i$ instead of treatment $j$?"* Because both terms share the same
baseline hazard $h_0(t)$, the difference **is** the personalised log-hazard
ratio between the two options:

- $\text{rec}_{ij}(x) > 0$ → treatment $i$ carries the **higher** risk for
  this individual → recommend $j$.
- $\text{rec}_{ij}(x) < 0$ → recommend $i$.

Our setting (equipment, per your brief): two **servicing regimes**, A and B,
and 1,000 machines. Regime B removes a specific failure mode that only
afflicts machines in a certain sensor region. So **the right regime genuinely
depends on the machine** — and that is exactly the situation the paper cares
about.
"""
)

st.header("1 · The proof: a linear Cox recommender is a CONSTANT")
st.markdown(
    r"""
Here is the paper's central argument for why you need a nonlinear model to do
this job. Take a standard linear Cox model with treatment as a covariate:

$$\hat h(x, \tau) = \beta^\top x + \gamma \tau
\qquad (\tau \in \{0, 1\}\ \text{is the treatment indicator})$$

Now compute the recommender function:

$$\text{rec}_{10}(x) = \hat h(x, 1) - \hat h(x, 0)
= \big(\beta^\top x + \gamma \cdot 1\big) - \big(\beta^\top x + \gamma \cdot 0\big)
= \gamma$$

**The $\beta^\top x$ terms cancel completely.** The recommendation is
$\gamma$ — *the same number for every individual*, regardless of its
covariates. A linear Cox model is therefore **structurally incapable** of
personalising a treatment decision: it can only ever say "treatment B is
better for everybody" or "treatment A is better for everybody". The $x$
literally cannot enter the answer.

Let's not take my algebra on trust — here it is done by a symbolic algebra
engine (SymPy), live:
"""
)

x1, x2, tau, g = sp.symbols("x_1 x_2 tau gamma")
b1, b2 = sp.symbols("beta_1 beta_2")
h_lin = b1 * x1 + b2 * x2 + g * tau
rec_lin = sp.simplify(h_lin.subs(tau, 1) - h_lin.subs(tau, 0))

st.markdown("**Linear Cox model:**")
st.latex(r"\hat h(x,\tau) = " + sp.latex(h_lin))
st.latex(r"\text{rec}_{10}(x) = \hat h(x,1) - \hat h(x,0) = "
         + sp.latex(rec_lin))
st.success(
    f"SymPy simplifies the difference to **{sp.latex(rec_lin)}** — the "
    "covariates have vanished entirely. Every machine gets the identical "
    "recommendation. **Verified, not asserted.**"
)

st.markdown(
    r"""
The classical statistician's fix is to add an **interaction term**, and it's
worth seeing that it *does* work — with a catch:
"""
)
d = sp.symbols("delta")
h_int = b1 * x1 + b2 * x2 + g * tau + d * tau * x1
rec_int = sp.simplify(h_int.subs(tau, 1) - h_int.subs(tau, 0))
st.markdown("**Cox with a hand-specified `treatment × x₁` interaction:**")
st.latex(r"\hat h(x,\tau) = " + sp.latex(h_int))
st.latex(r"\text{rec}_{10}(x) = " + sp.latex(rec_int))
st.info(
    f"Now the recommendation **{sp.latex(rec_int)}** *does* depend on $x_1$ — "
    "it varies across machines. So interactions rescue personalisation. "
    "**The catch:** you had to *know in advance* that the interaction was "
    "with $x_1$, and that it was linear. With 10 sensors there are 10 "
    "first-order interactions, 45 pairwise products, and infinitely many "
    "possible shapes. **DeepSurv learns whichever ones exist, without being "
    "told.** That is the paper's argument in a sentence."
)

st.header("2 · Both models, fitted for real, on machines that need different care")
st.markdown(
    """
The data (following the structure of the paper's treatment simulation): 1,000
machines, 10 sensors, half on regime A and half on regime B, assigned at
random. **Regime B eliminates a gaussian-shaped excess risk** centred on
sensors 0 and 1 — so B is a big win for machines near that centre, and
irrelevant for machines far from it.

Both models below are genuinely fitted, live: a linear Cox model (with
treatment as a covariate) and a DeepSurv network.
"""
)


@st.cache_data(show_spinner="Fitting Cox and training DeepSurv for real…")
def fit_both_live():
    return experiments.recommender_experiment()


res = artifacts.load("d6_recommender")
if st.session_state.get("_d6_live"):
    res = fit_both_live()
    d6_note = "just trained live on your machine"
elif res is not None:
    d6_note = artifacts.provenance(res)
else:
    res = fit_both_live()
    d6_note = "computed live"

cap, btn = st.columns([3, 1])
cap.caption(f"📦 A real Cox fit and a real DeepSurv fit — {d6_note}.")
if btn.button("▶ Re-train live (~5 s)"):
    st.session_state["_d6_live"] = True
    st.rerun()

X = np.asarray(res["X"])
T = np.asarray(res["T"])
E = np.asarray(res["E"])
trt = np.asarray(res["trt"])
gamma = float(res["gamma"])
rec_ds = np.asarray(res["rec_surface"])
rec_actual = np.asarray(res["rec_actual"])
agreed = np.asarray(res["agreed"]).astype(bool)

# the grid the recommendation surface was evaluated on (sensors 0 and 1)
gx, gy = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
rec_cox = np.full_like(rec_ds, gamma)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
truth = -np.log(10.0) * np.exp(-(gx ** 2 + gy ** 2) / (2 * 0.5 ** 2))
vmax = max(abs(truth).max(), abs(rec_ds).max(), abs(gamma)) or 1
for ax, data, title in [
    (axes[0], truth, "TRUTH: rec(x) = h(x,B) − h(x,A)"),
    (axes[1], rec_cox, f"linear Cox: rec(x) = γ = {gamma:.3f}\n"
                       "(the same everywhere!)"),
    (axes[2], rec_ds, "DeepSurv: rec(x) learned"),
]:
    im = ax.pcolormesh(gx, gy, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.contour(gx, gy, data, levels=[0], colors="black", linewidths=1.4)
    ax.set_xlabel("sensor 0")
    ax.set_ylabel("sensor 1")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    f"""
**Read the three panels — this is the paper's headline claim, reproduced.**

- **Left (the truth):** regime B strongly helps machines near the centre
  (deep blue: rec < 0 → give it B) and does nothing for machines at the edges
  (white: rec ≈ 0 → it doesn't matter). The right answer *depends on the
  machine*.
- **Middle (linear Cox):** a single flat colour. The fitted γ = **{gamma:.3f}**
  is the *only* thing it can say, to every machine, forever — exactly as the
  algebra predicted. It has averaged a machine-specific effect into one
  number, and in doing so has thrown away the entire question.
- **Right (DeepSurv):** recovers the shape. It was never told about sensors 0
  and 1, never told the effect was gaussian, never given an interaction term.
  It learned that regime B's benefit is concentrated in the centre, **from
  the data alone**.

That is the personalised treatment recommender, and it's why the model exists.
"""
)

st.header("3 · Does the recommendation actually help? Testing it")
st.markdown(
    "A recommendation is only worth something if following it is better than "
    "not. Here we split the *real* machines into those whose actual regime "
    "**agreed** with DeepSurv's recommendation and those where it **disagreed**, "
    "then compare their Kaplan–Meier survival curves — the same evaluation "
    "logic the paper uses."
)
from lifelines import KaplanMeierFitter  # noqa: E402

fig, ax = plt.subplots(figsize=(8, 3.8))
for mask, label, colour in [
        (agreed, f"regime matched the recommendation (n={agreed.sum()})",
         "#2e7d32"),
        (~agreed, f"regime went AGAINST it (n={(~agreed).sum()})", "#c62828")]:
    kmf = KaplanMeierFitter().fit(T[mask], E[mask], label=label)
    kmf.plot_survival_function(ax=ax, color=colour, ci_show=True)
ax.set_xlabel("months")
ax.set_ylabel("proportion still running")
ax.grid(alpha=0.25)
ax.set_title("machines whose servicing agreed with DeepSurv vs those that "
             "didn't", fontsize=10)
st.pyplot(fig)
plt.close(fig)

med_agree = float(res["median_agree"])
med_disagree = float(res["median_disagree"])
c1, c2 = st.columns(2)
c1.metric("median survival — recommendation followed", f"{med_agree:.2f} mo")
c2.metric("median survival — recommendation ignored", f"{med_disagree:.2f} mo",
          delta=f"{med_disagree - med_agree:+.2f} mo")
st.markdown(
    f"""
The machines that happened to receive the regime DeepSurv would have
recommended survived markedly longer (median **{med_agree:.2f}** vs
**{med_disagree:.2f}** months). Since regimes were assigned **at random**,
this gap is attributable to the recommendation itself rather than to
confounding — which is precisely why the paper runs this evaluation on
randomised data.

**Two honest caveats worth stating in a viva.** First, this is *our simulated*
data, where we built the interaction in ourselves — a friendly test. Second,
and more importantly: on **observational** (non-randomised) data this
evaluation is not valid, because whoever chose the treatment may have chosen
it *because* of the covariates. The whole apparatus of causal inference
(propensity scores, confounding, the potential-outcomes framework) then
becomes relevant. DeepSurv models the *conditional risk*, and a difference in
conditional risk is only a **causal** treatment effect under assumptions the
model itself cannot check.
"""
)

show_example(
    '''import sympy as sp

x1, x2, tau, gamma, beta1, beta2, delta = sp.symbols(
    "x1 x2 tau gamma beta1 beta2 delta")

# The linear Cox log-risk, with treatment as a covariate
h_linear = beta1 * x1 + beta2 * x2 + gamma * tau
rec_linear = sp.simplify(h_linear.subs(tau, 1) - h_linear.subs(tau, 0))
print("linear Cox   rec(x) =", rec_linear, "  <- no x! constant for everyone")

# Add a treatment-by-covariate interaction
h_inter = beta1 * x1 + beta2 * x2 + gamma * tau + delta * tau * x1
rec_inter = sp.simplify(h_inter.subs(tau, 1) - h_inter.subs(tau, 0))
print("with interaction rec(x) =", rec_inter, "  <- depends on x1")

# Proof it is constant: the derivative wrt every covariate is zero
print("d(rec_linear)/dx1 =", sp.diff(rec_linear, x1))
print("d(rec_linear)/dx2 =", sp.diff(rec_linear, x2))
print("d(rec_inter)/dx1  =", sp.diff(rec_inter, x1))''',
    """
- `sp.symbols(...)` — SymPy works with *symbols* rather than numbers, so it does algebra rather than arithmetic. This lets us prove a statement for *all* β and x, not merely check it for a few values.
- `h_linear.subs(tau, 1) - h_linear.subs(tau, 0)` — literally the recommender function: evaluate the log-risk under each treatment and subtract.
- `sp.simplify(...)` — returns `gamma`. **The covariates cancel; that's the proof.**
- `sp.diff(rec_linear, x1)` — the derivative of the recommendation with respect to a covariate is **exactly 0**, which is the formal statement of "the recommendation does not vary across individuals". For the interaction model, the derivative is `delta` — non-zero, so it *does* vary.
""",
)

guided_sandbox(
    key="d6",
    heavy=True,
    est="~10 s",
    steps="""
1. **Step 1** — with SymPy, define the linear Cox log-risk
   `h = beta1*x1 + beta2*x2 + gamma*tau` and compute
   `rec = h.subs(tau, 1) - h.subs(tau, 0)`, simplified. Print it. Confirm it's
   just `gamma`.
2. **Step 2** — prove it formally: print `sp.diff(rec, x1)` and
   `sp.diff(rec, x2)`. Both must be 0 — the recommendation is *constant in
   the covariates*.
3. **Step 3** — now add an interaction `delta*tau*x1` and redo steps 1–2. What
   is the recommendation now, and what is its derivative wrt `x1`?
4. **Step 4 (stretch)** — numerically, with the fitted models: for 5 randomly
   chosen machines, print the Cox recommendation (`cph.params_["trt"]`, the
   same for all 5) next to DeepSurv's `rec` (which will differ per machine).
   Seeing the identical column of numbers next to a varying one makes the
   point better than any algebra.
""",
    setup_code='''import numpy as np
import sympy as sp
import torch
from utils.mockdata import machines_treatment

x1, x2, tau, gamma, beta1, beta2, delta = sp.symbols(
    "x1 x2 tau gamma beta1 beta2 delta")

X, T, E, trt = machines_treatment(n=1000, seed=4)
print("1000 machines,", X.shape[1], "columns (10 sensors + treatment)")

# Step 1: the linear Cox recommender - show it simplifies to gamma


# Step 2: prove it: derivative wrt x1 and x2 are both zero


# Step 3: add delta*tau*x1 and redo


# Step 4 (stretch): compare Cox's constant vs DeepSurv's per-machine rec
#   (fit them yourself, or just reason about what the Cox column must look like)
''',
    solution_code='''import numpy as np
import pandas as pd
import sympy as sp
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from utils.mockdata import machines_treatment

x1, x2, tau, gamma, beta1, beta2, delta = sp.symbols(
    "x1 x2 tau gamma beta1 beta2 delta")

h = beta1 * x1 + beta2 * x2 + gamma * tau
rec = sp.simplify(h.subs(tau, 1) - h.subs(tau, 0))
print("linear Cox rec(x) =", rec)
print("d/dx1 =", sp.diff(rec, x1), "   d/dx2 =", sp.diff(rec, x2),
      "  -> constant for every machine")

h2 = h + delta * tau * x1
rec2 = sp.simplify(h2.subs(tau, 1) - h2.subs(tau, 0))
print("\\nwith interaction rec(x) =", rec2)
print("d/dx1 =", sp.diff(rec2, x1), "  -> now it varies with x1")

X, T, E, trt = machines_treatment(n=1000, seed=4)
cols = [f"x{i}" for i in range(10)] + ["trt"]
df = pd.DataFrame(X, columns=cols)
df["T"], df["E"] = T, E
cph = CoxPHFitter().fit(df, "T", "E")

mu, sd = X.mean(0), X.std(0)
torch.manual_seed(0)
net = nn.Sequential(
    nn.Linear(11, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
    nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
    nn.Linear(32, 1, bias=False))
opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Tt = torch.tensor(T, dtype=torch.float32)
Et = torch.tensor(E, dtype=torch.float32)
for _ in range(400):
    net.train(); opt.zero_grad()
    hh = net(Xs).squeeze(1)
    o = torch.argsort(Tt, descending=True)
    hs, Es = hh[o], Et[o]
    loss = -((hs - torch.logcumsumexp(hs, 0)) * Es).sum() / Es.sum()
    loss.backward(); opt.step()
net.eval()

def ds_rec(rows):
    g1, g0 = rows.copy(), rows.copy()
    g1[:, 10], g0[:, 10] = 1.0, 0.0
    with torch.no_grad():
        h1 = net(torch.tensor((g1 - mu) / sd, dtype=torch.float32)).squeeze(1)
        h0 = net(torch.tensor((g0 - mu) / sd, dtype=torch.float32)).squeeze(1)
    return (h1 - h0).numpy()

idx = [3, 17, 42, 88, 150]
recs = ds_rec(X[idx])
print(f"\\n{'machine':>8} {'sensor0':>8} {'sensor1':>8} {'Cox rec':>9} "
      f"{'DeepSurv rec':>13}")
for k, i in enumerate(idx):
    print(f"{i:>8} {X[i,0]:>8.2f} {X[i,1]:>8.2f} "
          f"{cph.params_['trt']:>9.3f} {recs[k]:>13.3f}")
print("\\nCox's column is IDENTICAL for every machine. DeepSurv's is not.")''',
)
