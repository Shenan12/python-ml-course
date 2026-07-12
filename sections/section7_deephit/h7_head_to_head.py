import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section7_deephit import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox

st.title("🥊 DeepSurv vs DeepHitSingle — the Honest Comparison")
st.markdown(
    """
Same cricket data. Same features. Same time-ordered split. Same tiny networks
(two 32-node layers). The only difference is the model. Let's see what actually
happens — and let's not decide the answer in advance.
"""
)

res = artifacts.load("h7_head_to_head")
if res is None:
    st.error("Run `python build_artifacts.py --only h7` to compute this page.")
    st.stop()

c_cox, c_ds, c_dh = (float(res["c_cox"]), float(res["c_ds"]),
                     float(res["c_dh"]))
ibs_ds, ibs_dh = float(res["ibs_ds"]), float(res["ibs_dh"])

st.header("1 · The scoreboard")
c1, c2, c3 = st.columns(3)
c1.metric("linear Cox — C-index", f"{c_cox:.4f}")
c2.metric("DeepSurv — C-td", f"{c_ds:.4f}", f"{c_ds - c_cox:+.4f} vs Cox")
c3.metric("DeepHitSingle — C-td", f"{c_dh:.4f}", f"{c_dh - c_ds:+.4f} vs DeepSurv")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].bar(["linear\nCox", "DeepSurv", "DeepHit\nSingle"],
            [c_cox, c_ds, c_dh],
            color=["#1565c0", "#c62828", "#6a1b9a"], edgecolor="white")
for i, v in enumerate([c_cox, c_ds, c_dh]):
    axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
axes[0].axhline(0.5, color="#9e9e9e", linestyle="--", linewidth=1)
axes[0].set_ylim(0.45, max(c_cox, c_ds, c_dh) + 0.03)
axes[0].set_ylabel("C-index (discrimination)")
axes[0].set_title("higher is better", fontsize=10)
axes[0].grid(alpha=0.25, axis="y")

axes[1].bar(["DeepSurv", "DeepHitSingle"], [ibs_ds, ibs_dh],
            color=["#c62828", "#6a1b9a"], edgecolor="white")
for i, v in enumerate([ibs_ds, ibs_dh]):
    axes[1].text(i, v + 0.0008, f"{v:.4f}", ha="center", fontsize=9)
axes[1].set_ylabel("Integrated Brier Score (calibration error)")
axes[1].set_title("lower is better", fontsize=10)
axes[1].grid(alpha=0.25, axis="y")
st.pyplot(fig)
plt.close(fig)
st.caption(f"📦 {artifacts.provenance(res)} — three real models on "
           f"{int(res['n_test']):,} held-out innings (2023–2026).")

st.header("2 · The verdict — a split decision, not a victory")
st.error(
    f"""
**The two models split the two metrics, and neither margin is meaningful.**

- **DeepHit wins discrimination** by **{c_dh - c_ds:+.4f}** C-td
  ({c_dh:.4f} vs {c_ds:.4f}).
- **DeepSurv wins calibration** by **{ibs_dh - ibs_ds:.4f}** IBS
  ({ibs_ds:.4f} vs {ibs_dh:.4f} — lower is better).

Both gaps are in the third decimal place. On {int(res['n_test']):,} test
innings, differences this small are **well inside the noise** — re-run with a
different random seed and they could swap places. The intellectually honest
summary is: **on this data, these two models perform the same.**

And that is *despite* our having **proved**, with a Schoenfeld test at
p < 0.0001, that proportional hazards is violated on exactly this data. The
theory said DeepHit should have an edge. It has, at best, a rounding error.

I could have buried this — tuned DeepHit harder, picked a friendlier split, or
quoted only the C-index and stayed quiet about the Brier score. **That is
precisely what you must not do.** Being able to *explain* a null result is worth
more in a viva than a win you can't account for.
"""
)

st.markdown(
    """
### Why doesn't the theory pay off?

Four honest explanations, in descending order of how much I believe them:

1. **The PH violation is statistically significant but practically small.** With
   25,000 innings, a Schoenfeld test will detect *any* departure from
   proportionality, however slight. Significance is not importance — a lesson
   you already know from every statistics course, and one that bites hard here.
   The hazard ratios shift, but not by enough to change *who ranks above whom*,
   and the C-index only cares about ranking.

2. **Discretisation throws away information.** DeepHit had to bin balls faced
   into 20 buckets. DeepSurv used the exact ball number. That's a real loss, and
   here it appears to roughly cancel out the gain from dropping the PH
   assumption.

3. **The features are weak.** *Every* model lands around 0.60. That smells like
   a ceiling imposed by the data, not the models. Cricket dismissals are
   genuinely, wonderfully random — a good ball can get anyone out. When all
   models are hitting the same wall, the wall is the data, and no architecture
   will save you. **Better features would beat a better model here.**

4. **DeepHit has more knobs and a small dataset.** Bins, scheme, α, σ — each is
   a chance to be slightly wrong, and each extra output neuron is a parameter to
   overfit with. Flexibility isn't free.
"""
)

st.info(
    f"""
**The result that actually matters from this section is not on this page.** It
was on the **Discretising Time** page.

- Switching DeepHit's bins from equidistant to quantile-based:
  **+0.09 C-index.**
- Switching architecture from DeepSurv to DeepHit:
  **{c_dh - c_ds:+.4f} C-index.**

**One preprocessing decision was ~30× more consequential than the choice of
architecture.** That is the most useful thing this entire section has to teach
you, and it generalises far beyond survival analysis: before you reach for a
newer model, make sure the one you have is being fed properly.
"""
)

st.header("3 · Calibration: are the curves telling the truth?")
st.markdown(
    "The C-index only checks *ordering*. This checks whether the predicted "
    "probabilities are **honest**: we average each model's predicted survival "
    "curve across all test batters and compare it against the Kaplan–Meier "
    "curve — the empirical truth."
)
fig, ax = plt.subplots(figsize=(9, 4))
ax.step(np.asarray(res["km_t"]), np.asarray(res["km_s"]), where="post",
        color="#37474f", linewidth=2.8, label="Kaplan–Meier (the truth)")
ax.plot(np.asarray(res["ds_index"]), np.asarray(res["ds_mean"]),
        color="#c62828", linewidth=2, label=f"DeepSurv (IBS {ibs_ds:.4f})")
ax.plot(np.asarray(res["dh_index"]), np.asarray(res["dh_mean"]),
        color="#6a1b9a", linewidth=2, linestyle="--",
        label=f"DeepHitSingle (IBS {ibs_dh:.4f})")
ax.set_xlabel("balls faced")
ax.set_ylabel("P(still batting)")
ax.set_xlim(0, 100)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.set_title("mean predicted survival vs the empirical truth", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Both models track the Kaplan–Meier curve closely — neither is *broken*. Look
for where they part company: DeepHit's curve is a **step function** (it can only
change value at a bin edge), while DeepSurv's is smooth. That stepping is
discretisation made visible, and it's part of why its Brier score is slightly
worse.
"""
)

with st.expander("🎓 Deeper statistics — what C-td and IBS actually compute "
                 "(the two metrics this page hangs on)"):
    st.markdown(
        r"""
**C-td — Antolini's time-dependent concordance.** Harrell's C-index needs
each subject reduced to *one* risk score, which is fine for Cox/DeepSurv
(the score never changes rank over time) but ill-posed for DeepHit — with
crossing curves, *who is riskier* depends on *when you ask*. Antolini's
fix: compare the pair **at the earlier failure's own time**. A comparable
pair $(i, j)$ with $T_i < T_j$ is concordant if

$$F_i(T_i \mid x_i) > F_j(T_i \mid x_j)$$

— at the moment $i$ was dismissed, the model gave $i$ the higher
probability of being out by then. For a PH model, $F_i(t) > F_j(t)$ at one
$t$ implies it at every $t$, so **C-td collapses back to Harrell's C** —
which is why comparing DeepSurv's C (from a score) with DeepHit's C-td
(from curves) on the scoreboard above is a fair fight and not a units
mismatch.

**IBS — the Brier score, censoring-corrected.** The Brier score at horizon
$t$ is a squared error on the survival *probability*:
$\mathbb{E}\big[(\mathbb{1}\{T > t\} - \hat S(t \mid x))^2\big]$ — a
**proper scoring rule**, minimised only by the true probability, which is
what earns it the word "calibration". Censoring breaks the naive average
(a not-out batter with $Y < t$ contributes an unknowable label), and the
standard repair (Graf et al. 1999) is **inverse-probability-of-censoring
weighting**: estimate the *censoring* survival curve $\hat G$ (a KM fit
with the roles of event and censoring swapped), then weight each usable
innings by $1/\hat G$ — dismissed-before-$t$ innings by
$1/\hat G(T_i)$, survivors by $1/\hat G(t)$. Each observed batter stands
in for the censored ones just like him — the same reweighting logic as
Horvitz–Thompson estimation in sampling theory. Integrating over $t$
gives the IBS. Two practical corollaries: IPCW needs its *own*
non-informative-censoring assumption (now about $C$), and $\hat G$ is
estimated on the test data — so IBS values are only comparable within the
same test set, never across datasets.

**And a bootstrap habit for the "third decimal" claim.** The statement
"these gaps are inside the noise" can be checked, not just asserted:
resample the test innings with replacement, recompute both metrics a few
hundred times, and look at the spread of the *difference*. If the
percentile interval covers zero comfortably, you can report the tie with
confidence instead of a hunch — a worthwhile check to run for any
head-to-head table in your dissertation.
"""
    )

st.header("4 · So when SHOULD you reach for DeepHit?")
st.markdown(
    """
Not "never" — the honest answer is "when the conditions favour it", and you can
now state those conditions precisely:

| Reach for **DeepHit** when… | Reach for **DeepSurv / Cox** when… |
|---|---|
| PH is violated **badly** — survival curves visibly **cross**, not merely drift | hazards are roughly proportional (test it!) |
| you have **competing risks** (multiple event types that censor each other) — this is DeepHit's home ground | there is one event type |
| you need **calibrated probabilities at specific times** and have enough data to bin finely | you need an interpretable **hazard ratio** to report |
| your dataset is **large** (bins and extra parameters need feeding) | your dataset is **small** — Cox's rigidity becomes a *virtue*, protecting you from overfitting |
| event times are **naturally discrete already** (balls, overs, months, cycles) | event times are continuous and precisely measured |

Notice the last row of that first column: our cricket data *is* naturally
discrete (you can't be dismissed on ball 20.4), which is one of the few points
genuinely in DeepHit's favour here — and it still wasn't enough.

### The sentence to take into your viva

> *DeepSurv relaxes Cox's **linearity** assumption but keeps **proportional
> hazards**. DeepHit abandons proportional hazards entirely by predicting a
> discrete-time probability distribution directly, and adds a ranking term to
> optimise concordance. Whether that freedom pays for its costs —
> discretisation, extra hyperparameters, lost interpretability — **is an
> empirical question about your dataset, not a theoretical victory.** On our ODI
> data, it did not.*

That is a far stronger thing to be able to say than "DeepHit is the newest so
it's the best."
"""
)

guided_sandbox(
    key="h7",
    heavy=True,
    est="~60 s",
    steps="""
1. **Step 1** — fit `DeepSurv` (pycox's `CoxPH`) on the cricket training data
   with the small `[32, 32]` net, then compute its test **C-td** and **IBS**
   using `EvalSurv`. (Remember `model.compute_baseline_hazards()` before
   predicting — DeepSurv needs it; DeepHit doesn't.)
2. **Step 2** — fit `DeepHitSingle` with the tuned settings (20 quantile bins,
   α=0.5, σ=0.5, lr=0.001) and compute the same two metrics.
3. **Step 3** — print both models' scores side by side. Do you reproduce this
   page's finding that they're roughly tied?
4. **Step 4 (stretch)** — the finding that matters: try to **beat both** by
   improving the *features* rather than the model. Add something cricket-aware
   — e.g. `balls_left_at_entry`, a `is_top_order` flag, or an interaction like
   `career_sr * is_chasing`. Does better feature engineering move the C-index
   more than swapping architectures did (±0.003)?
""",
    setup_code='''import numpy as np
import torch
import torchtuples as tt
import scipy.integrate as si
if not hasattr(si, "simps"):
    si.simps = si.simpson
from pycox.models import CoxPH, DeepHitSingle
from pycox.evaluation import EvalSurv
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)
grid = np.linspace(dte.min(), dte.max(), 50)
print(f"train {len(tr):,}  test {len(te):,}  features {len(cols)}")

# Step 1: DeepSurv - C-td and IBS


# Step 2: DeepHitSingle - C-td and IBS


# Step 3: print them side by side


# Step 4 (stretch): beat both with BETTER FEATURES
''',
    solution_code='''import numpy as np
import torch
import torchtuples as tt
import scipy.integrate as si
if not hasattr(si, "simps"):
    si.simps = si.simpson
from pycox.models import CoxPH, DeepHitSingle
from pycox.evaluation import EvalSurv
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)
grid = np.linspace(dte.min(), dte.max(), 50)

def score(surv):
    ev = EvalSurv(surv, dte, ete, censor_surv="km")
    return ev.concordance_td(), ev.integrated_brier_score(grid)

torch.manual_seed(0)
net = tt.practical.MLPVanilla(len(cols), [32, 32], 1, batch_norm=True,
                              dropout=0.1, output_bias=False)
ds = CoxPH(net, tt.optim.Adam(0.01))
ds.fit(Xtr, (dtr, etr), batch_size=256, epochs=100, verbose=False,
       val_data=(Xva, (dva, eva)),
       callbacks=[tt.callbacks.EarlyStopping(patience=15)])
ds.compute_baseline_hazards()          # DeepSurv needs this; DeepHit does not
c_ds, ibs_ds = score(ds.predict_surv_df(Xte))

lab = DeepHitSingle.label_transform(20, scheme="quantiles")
ytr = lab.fit_transform(dtr, etr)
yva = lab.transform(dva, eva)
torch.manual_seed(0)
net2 = tt.practical.MLPVanilla(len(cols), [32, 32], lab.out_features,
                               batch_norm=True, dropout=0.1)
dh = DeepHitSingle(net2, tt.optim.Adam(0.001), alpha=0.5, sigma=0.5,
                   duration_index=lab.cuts)
dh.fit(Xtr, ytr, batch_size=256, epochs=100, verbose=False,
       val_data=(Xva, yva),
       callbacks=[tt.callbacks.EarlyStopping(patience=15)])
c_dh, ibs_dh = score(dh.predict_surv_df(Xte))

print(f"{'model':16} {'C-td':>8} {'IBS':>8}")
print(f"{'DeepSurv':16} {c_ds:8.4f} {ibs_ds:8.4f}")
print(f"{'DeepHitSingle':16} {c_dh:8.4f} {ibs_dh:8.4f}")
print(f"\\ndifference: {c_dh - c_ds:+.4f} C-td  -> effectively a TIE")
print("Despite PH being violated (p < 0.0001). Theory does not guarantee")
print("an empirical win. Report what you measure, not what you expect.")''',
)
