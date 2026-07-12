import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section7_deephit import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🪓 Discretising Time into Bins")
st.markdown(
    r"""
DeepHit's first move is the one that makes everything else possible: **stop
treating time as continuous.**

Chop the time axis into $K$ bins, and ask the network a question it already
knows how to answer — a **classification** question:

> *"Of these $K$ bins, which one will the dismissal fall in?"*

A network that outputs $K$ numbers through a **softmax** gives a probability for
each bin: a **probability mass function** over when the event happens. No
hazard, no baseline, no proportionality — just a distribution, predicted
directly.

That's the whole architectural idea. But it hands you a new hyperparameter —
**where do the bin edges go?** — and this page exists because **I got it wrong
the first time, and it cost 0.09 of C-index.**
"""
)

res = artifacts.load("h3_binning")
if res is None:
    st.error("Run `python build_artifacts.py --only h3` to compute this page.")
    st.stop()

durations = np.asarray(res["durations"])

st.header("1 · The trap: balls faced are wildly skewed")
st.markdown(
    f"""
Look at the distribution of our durations. The median innings is
**{int(np.median(durations))} balls**, but the longest is
**{int(durations.max())}**. It is a classic long right tail — most batters get
out quickly, a few bat for hours.
"""
)

scheme = st.radio(
    "how should we place the 20 bin edges?",
    ["equidistant — cut [0, max] into 20 equal-width slices",
     "quantiles — cut so each bin holds an equal number of events"],
    index=1,
)
key = "cuts_equidistant" if scheme.startswith("equidistant") else "cuts_quantiles"
cuts = np.asarray(res[key])

fig, ax = plt.subplots(figsize=(9.5, 3.6))
ax.hist(durations, bins=60, color="#90a4ae", edgecolor="white")
for c in cuts:
    ax.axvline(c, color="#c62828", linewidth=1.3, alpha=0.85)
ax.set_xlabel("balls faced")
ax.set_ylabel("innings")
ax.set_title(f"{len(cuts)} bin edges — {scheme.split(' —')[0]}", fontsize=10)
st.pyplot(fig)
plt.close(fig)

counts, _ = np.histogram(durations, bins=np.r_[0, cuts[1:], durations.max() + 1])
fig, ax = plt.subplots(figsize=(9.5, 2.4))
ax.bar(range(len(counts)), counts, color="#1565c0", edgecolor="white")
ax.set_xlabel("bin index")
ax.set_ylabel("innings in bin")
ax.set_title("how many innings land in each bin?", fontsize=10)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

empty = int((counts == 0).sum())
biggest = int(counts.max())
if scheme.startswith("equidistant"):
    st.error(
        f"""
**This is the bug.** Equal-width bins over a skewed variable are a disaster:
the busiest bin holds **{biggest:,} innings** while **{empty} bins are
completely empty** (or nearly so).

Think about what that does to the model. The first bin swallows a huge chunk of
all dismissals — so a batter dismissed on ball 2 and a batter dismissed on ball
8 land in the *same* bin and become **indistinguishable**. The model is
forbidden from separating them, no matter how good its features. Meanwhile the
far bins have almost no data to learn from, so their probabilities are noise.

You have thrown away most of your resolution exactly where all your data is.
"""
    )
else:
    st.success(
        f"""
**Quantile bins fix it.** Each bin now holds a comparable number of innings
(biggest bin: **{biggest:,}**, empty bins: **{empty}**). The bins are *narrow*
where the data is dense (a bin might span balls 3–5) and *wide* out in the tail
(one bin covers everything past ~70 balls).

That's exactly the right allocation: **spend your resolution where the events
actually are.**
"""
    )

st.header("2 · What it costs: the same model, both binnings")
st.markdown(
    "Every number below is a real DeepHitSingle model, trained on the cricket "
    "data with **identical** architecture, features and hyperparameters. "
    "**Only the bin placement differs.**"
)
results = pd.read_csv(io.StringIO(res["results_csv"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, metric, better in [(axes[0], "C-td", "higher is better"),
                           (axes[1], "IBS", "lower is better")]:
    for sch, colour, marker in [("equidistant", "#c62828", "o"),
                                ("quantiles", "#2e7d32", "s")]:
        sub = results[results.scheme == sch]
        ax.plot(sub["num_bins"], sub[metric], marker=marker, color=colour,
                linewidth=2, label=sch)
    ax.set_xlabel("number of bins")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} ({better})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if metric == "C-td":
        ax.axhline(0.5, color="#9e9e9e", linestyle="--", linewidth=1)
        ax.text(6, 0.505, "coin flip", fontsize=7, color="#757575")
st.pyplot(fig)
plt.close(fig)

st.dataframe(
    results[["scheme", "num_bins", "C-td", "IBS"]]
    .style.format({"C-td": "{:.4f}", "IBS": "{:.4f}"})
    .highlight_max(subset=["C-td"], color="#c8e6c9")
    .highlight_min(subset=["IBS"], color="#c8e6c9"),
    hide_index=True, width="stretch")

eq20 = results[(results.scheme == "equidistant") & (results.num_bins == 20)]
qt20 = results[(results.scheme == "quantiles") & (results.num_bins == 20)]
gap = float(qt20["C-td"].iloc[0]) - float(eq20["C-td"].iloc[0])
st.error(
    f"""
**At 20 bins, switching from equidistant to quantile bins is worth
+{gap:.3f} C-index** ({float(eq20['C-td'].iloc[0]):.3f} →
{float(qt20['C-td'].iloc[0]):.3f}).

To put that in perspective: **DeepSurv's entire advantage over a linear Cox
model, on every real dataset in the DeepSurv paper, was between +0.013 and
+0.045.** This one preprocessing choice — a single keyword argument — matters
several times more than the difference between the two published architectures.

**This is the single most useful thing on this page.** When a paper reports
"model X beat model Y", your first question should be *"was Y tuned as
carefully as X?"*. An equidistant-binned DeepHit looks like a broken model. It
isn't. It's a fine model, badly prepared.
"""
)

st.header("3 · How many bins?")
st.markdown(
    f"""
Read the left-hand chart again, along the **quantiles** line. The trade-off:

- **Too few bins** (5): everything is coarse. A batter dismissed on ball 3 and
  one dismissed on ball 15 look identical to the model, so it cannot rank them
  — the C-index suffers.
- **Too many bins**: each bin holds fewer events, so each of the $K$ softmax
  outputs is estimated from less data and gets noisy. You also add parameters
  (the output layer is $K$ wide).
- **The sweet spot** for this data sits around **20** bins.

This is the bias–variance trade-off from Section 2 wearing yet another costume —
coarse bins = high bias, many bins = high variance. And as always, you pick the
number on the **validation** set, not the test set.
"""
)

with st.expander("🎓 Deeper statistics — discrete-time survival is older than "
                 "you think, and better-behaved than it looks"):
    st.markdown(
        r"""
Discretising time is not a deep-learning hack — it's the **life table**,
the oldest object in survival analysis (actuaries have binned lifetimes
into years since the 1600s). The discrete-time framework has its own clean
calculus, and it's worth having:

- **Discrete hazard:** $h_k = P(T = k \mid T \ge k)$ — the chance the
  dismissal falls in bin $k$ given the batter reached it. Unlike the
  continuous hazard, this *is* a probability.
- **Survival:** $S(k) = P(T > k) = \prod_{j \le k} (1 - h_j)$ — survive
  each bin in turn. (Recognise the form? The Kaplan–Meier estimator is
  exactly this with $\hat h_j = d_j/n_j$ — KM *is* discrete-time survival
  analysis where every observed failure time is its own bin.)
- **PMF:** $p_k = P(T = k) = h_k \prod_{j<k}(1 - h_j)$ — fail in bin $k$ =
  survive $k\!-\!1$ bins, then fail. Geometric-distribution logic with a
  different $h$ each step.

These identities mean "predict the PMF" (DeepHit's choice) and "predict the
hazards" (the classical *discrete-time logistic hazard* model, and pycox's
`LogisticHazard`) are two parametrisations of the same object — you can
always convert one into the other. DeepHit's softmax predicts all the
$p_k$ jointly; a hazard model predicts $K$ conditional Bernoullis. Neither
assumes proportionality — discretisation alone already buys you freedom
from PH.

**And the quantile trick is one you already know.** Placing bin edges at
quantiles of the event distribution is the **probability integral
transform** in action: if the edges sit at the
$\tfrac{1}{K}, \tfrac{2}{K}, \dots$ quantiles of $T$, then the bin index of an event is
(approximately) uniform — every class balanced by construction. It's the
same reasoning as histogram equalisation, or using ranks instead of raw
values: statistics on a skewed variable behave better after a monotone map
to uniformity, and no information the model needs is lost because the map
is invertible on the bins.
"""
    )

st.header("4 · In code")
show_example(
    experiments.SNIPPETS["h3_binning"],
    """
- `DeepHitSingle.label_transform(20, scheme=...)` — pycox's discretiser. `scheme="quantiles"` places the cuts at equally-spaced *quantiles of the event times*; `"equidistant"` places them at equal *widths*. **This one argument is the whole page.**
- `labtrans.fit_transform(dtr, etr)` — note it is fitted on the **training durations only** (the anti-leakage rule: your bin edges are learned parameters too, and must not see the test set), then `transform` is applied to validation.
- `labtrans.out_features` — the number of output neurons the network needs: one per bin. The architecture literally depends on your discretisation choice.
- `y_train` is a `(bin_index, event)` pair per innings — the duration has become a **class label**. That's the conceptual leap: survival prediction is now classification over bins.
- `EvalSurv(...).concordance_td()` — the **time-dependent** concordance index (Antolini's C-index). Because DeepHit predicts a full survival curve rather than a single risk score, the appropriate concordance measure compares curves at each event time. It is the standard metric in the DeepHit literature.
""",
    key="h3_binning",
    heavy=True,
    est="~30 s",
)

guided_sandbox(
    key="h3",
    heavy=True,
    est="~90 s",
    steps="""
1. **Step 1** — print the quantiles of `balls_faced` (try `np.percentile(dtr,
   [10, 25, 50, 75, 90, 99])`). Confirm for yourself how skewed it is.
2. **Step 2** — build both discretisers,
   `DeepHitSingle.label_transform(10, scheme="equidistant")` and the same with
   `"quantiles"`, `fit` each on the training durations, and print their
   `.cuts`. Compare the two sets of bin edges — where does each spend its
   resolution?
3. **Step 3** — for each scheme, use `np.histogram(dtr, bins=...)` to count how
   many innings land in each bin. Print the counts. How many equidistant bins
   are (nearly) empty?
4. **Step 4 (stretch)** — train a DeepHitSingle with each scheme (10 bins,
   100 epochs) and print both test C-td scores. You should reproduce the gap
   this page reports. **This is the experiment that proves the point — worth
   running once yourself.**
""",
    setup_code='''import numpy as np
import torch
import torchtuples as tt
import scipy.integrate as si
if not hasattr(si, "simps"):
    si.simps = si.simpson
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)
print("durations:", dtr.min(), "-", dtr.max(), " median", np.median(dtr))

# Step 1: the quantiles of balls faced


# Step 2: build both discretisers, print their .cuts


# Step 3: count innings per bin under each scheme


# Step 4 (stretch): train both, compare C-td
''',
    solution_code='''import numpy as np
import torch
import torchtuples as tt
import scipy.integrate as si
if not hasattr(si, "simps"):
    si.simps = si.simpson
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)

print("percentiles of balls faced:")
for p in [10, 25, 50, 75, 90, 99]:
    print(f"  {p:2d}th: {np.percentile(dtr, p):5.0f}")
print("-> heavily right-skewed. Equal-width bins will be a disaster.\\n")

for scheme in ["equidistant", "quantiles"]:
    lt = DeepHitSingle.label_transform(10, scheme=scheme)
    lt.fit(dtr, etr)
    print(f"{scheme:12} cuts: {np.round(lt.cuts, 1)}")
    counts, _ = np.histogram(dtr, bins=np.r_[0, lt.cuts[1:], dtr.max() + 1])
    print(f"{'':12} innings per bin: {counts}")
    print(f"{'':12} empty/near-empty bins: {(counts < 50).sum()}\\n")

for scheme in ["equidistant", "quantiles"]:
    lt = DeepHitSingle.label_transform(10, scheme=scheme)
    ytr = lt.fit_transform(dtr, etr)
    yva = lt.transform(dva, eva)
    torch.manual_seed(0)
    net = tt.practical.MLPVanilla(len(cols), [32, 32], lt.out_features,
                                  batch_norm=True, dropout=0.1)
    m = DeepHitSingle(net, tt.optim.Adam(0.001), alpha=0.5, sigma=0.5,
                      duration_index=lt.cuts)
    m.fit(Xtr, ytr, batch_size=256, epochs=100, verbose=False,
          val_data=(Xva, yva),
          callbacks=[tt.callbacks.EarlyStopping(patience=15)])
    ev = EvalSurv(m.predict_surv_df(Xte), dte, ete, censor_surv="km")
    print(f"{scheme:12}: test C-td {ev.concordance_td():.4f}")
print("\\nSame model. Same features. Only the bin edges moved.")''',
)
