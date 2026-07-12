import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section6_deepsurv import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🔬 The Paper's Experiment, Rebuilt")
st.markdown(
    r"""
This is the experiment that justifies the whole model, and you can now run it
yourself. Katzman et al. simulate survival data where **they control the true
risk function**, then ask: does DeepSurv beat Cox? The answer is a beautifully
sharp *"it depends — and here is exactly what it depends on."*

Their two scenarios (we reproduce both, with machines instead of patients):

| | true log-risk h(x) | what it means |
|---|---|---|
| **Linear** | $h(x) = x_0 + 2x_1$ | risk rises steadily with two sensors — *exactly the shape Cox assumes* |
| **Nonlinear (gaussian)** | $h(x) = \log(\lambda_{max})\exp\!\big(\!-\frac{x_0^2 + x_1^2}{2r^2}\big)$ | risk peaks in a **circular region** — machines fail when both sensors sit near zero, and are safe at *either* extreme |

with $\lambda_{max} = 5$, $r = 0.5$. In both cases there are **10 sensors but
only 2 matter**, values drawn Uniform[−1, 1), failure times exponential with
rate $e^{h(x)}$, and **50% of machines randomly right-censored**.

**Flip the switch below and watch Cox collapse.**
"""
)

risk_type = st.radio(
    "the TRUE risk function generating the data",
    ["linear  —  h(x) = x₀ + 2x₁",
     "nonlinear (gaussian)  —  h(x) = log(λ)·exp(−(x₀²+x₁²)/2r²)"],
    index=1,
)
risk = "linear" if risk_type.startswith("linear") else "nonlinear"

c1, c2 = st.columns(2)
n_train = c1.select_slider("training machines", experiments.D7_TRAIN_SIZES,
                           value=2000)
early_stop = c2.toggle(
    "early stopping (halt when validation loss stops improving)", value=True,
    help="Off = train the full 512 epochs and watch DeepSurv overfit.")

st.header("1 · What the true risk surface looks like")
gx, gy = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 120))
if risk == "linear":
    true_h = gx + 2 * gy
else:
    true_h = np.log(5.0) * np.exp(-(gx ** 2 + gy ** 2) / (2 * 0.5 ** 2))

fig, ax = plt.subplots(figsize=(5.6, 4.4))
im = ax.pcolormesh(gx, gy, true_h, cmap="magma", shading="auto")
fig.colorbar(im, ax=ax, label="true log-risk h(x)")
ax.contour(gx, gy, true_h, levels=8, colors="white", linewidths=0.5,
           alpha=0.5)
ax.set_xlabel("sensor 0")
ax.set_ylabel("sensor 1")
ax.set_aspect("equal")
ax.set_title("the truth we are asking each model to recover", fontsize=10)
st.pyplot(fig)
plt.close(fig)

if risk == "linear":
    st.info(
        "**A plane.** Risk climbs steadily as the sensors rise. This is "
        "*precisely* the functional form the Cox model assumes — Cox is the "
        "**correctly specified** model here. A flexible model can, at best, "
        "match it."
    )
else:
    st.warning(
        "**A hill.** Risk is highest at the centre and falls away in every "
        "direction. Notice what this does to a *linear* model: as sensor 0 "
        "goes from −1 to 0, risk **rises**; from 0 to +1, it **falls**. The "
        "best-fitting straight line through that relationship is **flat** — "
        "so Cox will estimate β ≈ 0 and conclude the sensors carry no "
        "information at all. It cannot merely do *worse* here; it is "
        "structurally blind."
    )


@st.cache_data(show_spinner="Fitting Cox and training DeepSurv for real — "
                            "this takes ~20 s…")
def run_experiment_live(risk, n_train, early_stop):
    """Only ever called when you press ▶ Train — see the note below."""
    return experiments.simulation_experiment(risk, n_train, early_stop)


art_name = f"d7_{risk}_{n_train}_{'es' if early_stop else 'full'}"
result = artifacts.load(art_name)
live_key = f"_d7_live_{art_name}"

if st.session_state.get(live_key):
    result = run_experiment_live(risk, n_train, early_stop)
    source_note = "just trained live on your machine"
elif result is not None:
    source_note = (f"{artifacts.provenance(result)} — a real fit, saved by "
                   "`build_artifacts.py` so the page loads instantly")
else:
    st.warning(
        f"No precomputed result for this combination yet "
        f"(`{art_name}`). Press the button to train it now (~20 s), or run "
        "`python build_artifacts.py` once to precompute every combination."
    )
    if st.button("▶ Train this configuration now (~20 s)",
                 key=f"btn_{art_name}"):
        st.session_state[live_key] = True
        st.rerun()
    st.stop()

epochs_run = int(result.get("epochs_run", 0))
max_epochs = int(result.get("max_epochs", experiments.D7_MAX_EPOCHS))

c_cox = float(result["c_cox"])
c_ds = float(result["c_ds"])
cox_surf = np.asarray(result["cox_surface"])
ds_surf = np.asarray(result["ds_surface"])
betas = np.asarray(result["betas"])
Tte = np.asarray(result["Tte"])
Ete = np.asarray(result["Ete"])
cox_risk = np.asarray(result["cox_risk"])
ds_risk = np.asarray(result["ds_risk"])

st.header("2 · What each model learned")
cap, btn = st.columns([3, 1])
cap.caption(
    f"📦 Results {source_note}. Every number below came from genuinely "
    "training these models — press ▶ to watch it happen again."
)
if btn.button("▶ Re-train live (~20 s)", key=f"retrain_{art_name}"):
    st.session_state[live_key] = True
    st.rerun()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, surf, name, c in [(axes[0], cox_surf, "linear Cox", c_cox),
                          (axes[1], ds_surf, "DeepSurv", c_ds)]:
    im = ax.pcolormesh(gx, gy, surf, cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.contour(gx, gy, surf, levels=8, colors="white", linewidths=0.5,
               alpha=0.5)
    ax.set_xlabel("sensor 0")
    ax.set_ylabel("sensor 1")
    ax.set_aspect("equal")
    ax.set_title(f"{name} — test C-index {c:.3f}", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3, c4 = st.columns(4)
c1.metric("linear Cox — C-index", f"{c_cox:.3f}")
c2.metric("DeepSurv — C-index", f"{c_ds:.3f}",
          delta=f"{c_ds - c_cox:+.3f} vs Cox")
c3.metric("random guessing", "0.500")
c4.metric("epochs actually trained", epochs_run,
          delta="early stop" if early_stop else f"full {max_epochs}",
          delta_color="off")

with st.expander("📉 The training curves — and what early stopping is doing"):
    tr = np.asarray(result["train_curve"], dtype=float)
    va = np.asarray(result["val_curve"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(tr, color="#1565c0", linewidth=1.8, label="training loss")
    ax.plot(va, color="#e65100", linewidth=1.8, label="validation loss")
    best_epoch = int(np.argmin(va))
    ax.axvline(best_epoch, color="#2e7d32", linestyle=":", linewidth=1.6,
               label=f"best validation loss (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Cox negative log partial likelihood")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if early_stop:
        st.markdown(
            f"""
Early stopping halted training at **epoch {epochs_run}** and kept the weights
from the best validation epoch. Notice the validation curve had already
flattened — everything after that point would have been the network learning
the training set's *noise*.
"""
        )
    else:
        st.markdown(
            f"""
With early stopping **off**, the network ran all **{epochs_run}** epochs. Look
at the gap that opens up: the training loss keeps falling while the validation
loss bottoms out (around epoch {best_epoch}) and then **rises**. That widening
gap is overfitting, live — the exact U-curve from Section 2, now on a survival
model.

Toggle early stopping back on and compare the C-index: on this data, letting it
run to the end makes the model **worse**, not better.
"""
        )

st.markdown("**The paper's published numbers for this same experiment:**")
st.markdown(
    f"""
| | Cox (CPH) | DeepSurv | our run just now |
|---|---|---|---|
| **Simulated linear** | 0.779 | 0.778 | {"**Cox " + f"{c_cox:.3f}" + " / DeepSurv " + f"{c_ds:.3f}**" if risk == "linear" else "— (switch the toggle)"} |
| **Simulated nonlinear** | 0.487 | 0.652 | {"**Cox " + f"{c_cox:.3f}" + " / DeepSurv " + f"{c_ds:.3f}**" if risk == "nonlinear" else "— (switch the toggle)"} |
"""
)

if risk == "linear":
    st.success(
        f"""
**The honest result — and the one students always skip.** On linear data,
Cox ({c_cox:.3f}) and DeepSurv ({c_ds:.3f}) are **essentially tied**, just as
the paper found (0.779 vs 0.778). And so they should be: Cox is the correctly
specified model, DeepSurv is a strictly more flexible one, and a flexible
model's best case is to *rediscover* the simple truth — while paying for the
privilege in variance.

Look at the left panel: Cox's learned surface is a plane, and it's the *right*
plane. Right panel: DeepSurv has learned an approximately planar surface too
— it found the linear truth without being told the truth was linear.

**Never tell anyone "DeepSurv beats Cox."** Tell them: *DeepSurv matches Cox
when Cox is right, and beats it when Cox is wrong.* That is a statement about
model misspecification, and it's a far more defensible claim.
"""
    )
else:
    noise_max = int(np.argmax(np.abs(betas[2:]))) + 2
    st.error(
        f"""
**Cox has collapsed to {c_cox:.3f} — a coin flip.** (The paper reports 0.487;
we get {c_cox:.3f}.) Look at the left panel: Cox's best plane through a
circular hill is nearly **flat** — its fitted coefficients on the two genuinely
informative sensors are β₀ = {betas[0]:+.3f}, β₁ = {betas[1]:+.3f}, both
crushed to near-nothing. Averaged over the whole sensor range, the rise and
the fall of the hill cancel, and a linear term has *nothing left to fit*.

**It gets worse, and this is the part worth remembering.** Cox doesn't merely
fail to find the signal — it has assigned a coefficient of
{betas[noise_max]:+.3f} to **sensor {noise_max}, which is pure noise** and has
no effect on failure whatsoever. On a run like this, the noise sensor can
easily come out with a *smaller p-value than the real ones*. A practitioner
reading this output would conclude that sensor {noise_max} drives failure and
that sensors 0 and 1 are irrelevant — the **exact inverse of the truth**, stated
with confidence intervals.

DeepSurv ({c_ds:.3f}) recovers the hill (right panel) and ranks machines far
better. And note precisely where the blame lies: not with Cox being a *weak*
model — Cox is a superb model — but with **misspecification**. It was asked a
question its functional form cannot answer, and it answered anyway.
"""
    )

with st.expander("🎓 Deeper statistics — reading this page like a "
                 "simulation-study methodologist"):
    st.markdown(
        r"""
A simulation study is the statistician's controlled experiment, and it has
the same anatomy every time: a known **data-generating process** (the truth
above), an **estimand** (here: the risk *ordering*, scored by C-index),
competing **estimators** (Cox, DeepSurv), and **replication** to separate
signal from Monte Carlo noise. Three habits worth taking from this page
into your dissertation:

- **Our numbers differ from the paper's in the second decimal, and they
  should.** A C-index computed on one simulated test set is itself a random
  variable — different seed, different draw, different value. What must
  *reproduce* is the qualitative structure (tie under linear, collapse vs
  ~0.65 under nonlinear), not the third decimal. When you see a paper
  report 0.487, read it as "indistinguishable from 0.5", not as a constant
  of nature.
- **Where Cox's β lands is not random garbage — it's a theorem.** Under
  misspecification, the MLE converges to the parameter of the *best
  approximation within the model class* (the KL-projection of the truth
  onto the model — White's quasi-MLE result). For a symmetric hill on
  Uniform[−1,1) sensors, the best linear approximation is flat, so
  $\hat\beta \to 0$ *with* ever-tighter confidence intervals as $n$ grows.
  The confidence intervals are honestly reporting uncertainty about the
  wrong quantity. Misspecification is not detectable from standard errors.
- **The noise-sensor false positive is multiplicity, not misfortune.** With
  8 pure-noise coefficients tested at the 5% level, the chance at least one
  looks "significant" approaches $1 - 0.95^8 \approx 34\%$ per run. Combine
  that with the real signal being invisible to the model, and the most
  significant coefficient in the table is *more likely to be noise than
  truth* — a small, vivid case for multiple-testing discipline whenever you
  read a coefficient table.
"""
    )

st.header("3 · The C-index, seen as a ranking")
st.markdown(
    "The C-index is abstract; here is what it *means*. We take the 1,000 test "
    "machines, sort them by each model's predicted risk, and plot their actual "
    "observed failure times. A good model produces a downward slope — "
    "high-risk machines (left) fail early."
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, r, name, c in [(axes[0], cox_risk, "linear Cox", c_cox),
                       (axes[1], ds_risk, "DeepSurv", c_ds)]:
    order = np.argsort(-r)                     # riskiest first
    obs = Tte[order]
    ev = Ete[order].astype(bool)
    ax.scatter(np.arange(len(obs))[ev], obs[ev], s=8, color="#c62828",
               alpha=0.6, label="observed failure")
    ax.scatter(np.arange(len(obs))[~ev], obs[~ev], s=8, color="#90a4ae",
               alpha=0.4, marker=">", label="censored")
    # rolling mean of failure times, to show the trend
    fails = np.where(ev)[0]
    if len(fails) > 40:
        w = 40
        roll = pd.Series(obs[ev]).rolling(w, center=True).mean()
        ax.plot(fails, roll, color="#1565c0", linewidth=2.4,
                label=f"trend (window {w})")
    ax.set_xlabel("machines, sorted by predicted risk (riskiest → safest)")
    ax.set_ylabel("actual time to failure")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(f"{name} — C-index {c:.3f}", fontsize=10)
    ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(
    "A model with a real C-index shows a clear upward trend left-to-right "
    "(the machines it called risky did fail sooner). A model at C ≈ 0.5 shows "
    "a **flat cloud** — its ordering carries no information about who fails "
    "first. Compare the two panels under the nonlinear setting and the "
    "difference is unmistakable."
)

st.header("4 · The experiment in code")
show_example(
    experiments.SNIPPETS["d7_example"],
    """
- `machines(risk=...)` — our generator, implementing the paper's two risk functions exactly (see `utils/mockdata.py`).
- `f = lambda A: ((A - mu) / sd)` — standardization using **training** statistics, applied to every split. A one-line lambda (Functions page) doing anti-leakage duty.
- `CoxPHFitter(penalizer=0.01)` — a small ridge penalty; with 10 covariates and 50% censoring, the unpenalised fit can be numerically unstable, and lifelines will warn you. This is L2 regularization on a *classical* model — the same idea as the network's weight decay.
- `tt.practical.MLPVanilla(10, [32, 32], 1, batch_norm=True, dropout=0.1, output_bias=False)` — torchtuples' shortcut for the exact DeepSurv architecture: 10 inputs → two 32-node hidden layers with BatchNorm and dropout → **1 output with no bias** (unidentifiable, as we proved).
- `CoxPH(net, tt.optim.Adam(0.01))` — pycox's DeepSurv. Its loss is the Cox negative log partial likelihood you implemented from scratch two pages ago; you know exactly what it's doing.
- Watch the printed `beta_0`/`beta_1` flip between the two runs: substantial under `linear`, ≈ 0 under `nonlinear`. That's the misspecification, visible in the coefficients themselves.
""",
    key="d7_example",
    heavy=True,
    est="~35 s",
)

guided_sandbox(
    key="d7",
    heavy=True,
    est="~40 s",
    steps="""
1. **Step 1** — generate nonlinear training and test data with `machines(...)`,
   standardize using **training** mean/std only, and fit a `CoxPHFitter`
   (use `penalizer=0.01`). Print its C-index on the test set — remember the
   minus sign: `concordance_index(Tte, -risk_scores, Ete)`.
2. **Step 2** — print the **full** coefficient table (`cph.summary[["coef",
   "p"]]`). Two things to look for: (a) how tiny the coefficients on the
   genuinely informative `x0`/`x1` are, and (b) whether any of the eight
   **pure-noise** sensors `x2`…`x9` has picked up a *smaller* p-value than
   they did. Cox is not just blind here — it can actively point at the wrong
   sensor.
3. **Step 3** — train DeepSurv with pycox (`MLPVanilla` + `CoxPH`, 300 epochs)
   and print its test C-index. How big is the gap?
4. **Step 4 (stretch)** — the fair-fight test. Re-run Cox but **give it the
   right features**: add `x0**2` and `x1**2` as extra columns (i.e. tell it
   the truth is quadratic). Does Cox catch up with DeepSurv? What does this
   tell you about *when* you actually need a neural network — and what it
   would cost you to guess wrong on a real dataset with 40 covariates?
""",
    setup_code='''import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

Xtr, Ttr, Etr = machines(n=2000, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=1000, risk="nonlinear", seed=2)
mu, sd = Xtr.mean(0), Xtr.std(0)
f = lambda A: ((A - mu) / sd).astype("float32")
cols = [f"x{i}" for i in range(10)]
print("data ready:", Xtr.shape, "train,", Xte.shape, "test")

# Step 1: fit CoxPHFitter, print test C-index


# Step 2: the coefficients and p-values on x0 and x1


# Step 3: train DeepSurv, print test C-index


# Step 4 (stretch): give Cox the squared terms and see it catch up
''',
    solution_code='''import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

Xtr, Ttr, Etr = machines(n=2000, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=1000, risk="nonlinear", seed=2)
mu, sd = Xtr.mean(0), Xtr.std(0)
f = lambda A: ((A - mu) / sd).astype("float32")
cols = [f"x{i}" for i in range(10)]

df = pd.DataFrame(f(Xtr), columns=cols)
df["T"], df["E"] = Ttr, Etr
cph = CoxPHFitter(penalizer=0.01).fit(df, "T", "E")
c_cox = concordance_index(Tte, -(f(Xte) @ cph.params_.values), Ete)
print(f"Cox test C-index: {c_cox:.4f}   (a coin flip is 0.5)")

print("\\nCox's full verdict (x0, x1 are the ONLY sensors that matter;")
print("x2..x9 are pure noise):")
print(cph.summary[["coef", "p"]].round(4).to_string())
informative = np.abs(cph.params_[["x0", "x1"]]).mean()
noise = np.abs(cph.params_[[f"x{i}" for i in range(2, 10)]]).mean()
print(f"\\nmean |coef| on the 2 REAL sensors:  {informative:.4f}")
print(f"mean |coef| on the 8 NOISE sensors: {noise:.4f}")
print("-> barely distinguishable. Check the p-value column: a noise sensor")
print("   may well be 'more significant' than the real ones. Cox is not")
print("   just failing to find the signal - it is pointing at the wrong one.")

torch.manual_seed(0)
net = tt.practical.MLPVanilla(10, [32, 32], 1, batch_norm=True, dropout=0.1,
                              output_bias=False)
model = CoxPH(net, tt.optim.Adam(0.01))
model.fit(f(Xtr), (Ttr.astype("float32"), Etr.astype("float32")),
          batch_size=256, epochs=300, verbose=False)
c_ds = concordance_index(Tte, -model.predict(f(Xte)).ravel(), Ete)
print(f"\\nDeepSurv test C-index: {c_ds:.4f}  (+{c_ds - c_cox:.3f} over Cox)")

Xtr2 = np.column_stack([f(Xtr), f(Xtr)[:, 0] ** 2, f(Xtr)[:, 1] ** 2])
Xte2 = np.column_stack([f(Xte), f(Xte)[:, 0] ** 2, f(Xte)[:, 1] ** 2])
df2 = pd.DataFrame(Xtr2, columns=cols + ["x0_sq", "x1_sq"])
df2["T"], df2["E"] = Ttr, Etr
cph2 = CoxPHFitter(penalizer=0.01).fit(df2, "T", "E")
c_cox2 = concordance_index(Tte, -(Xte2 @ cph2.params_.values), Ete)
print(f"\\nCox WITH the squared terms: {c_cox2:.4f}  <- it catches up!")
print("Lesson: Cox was never weak - it was misspecified. But you only knew")
print("to add x0^2 and x1^2 because I TOLD you the truth was gaussian.")
print("With 40 real covariates and no oracle, DeepSurv finds it for you.")''',
)
