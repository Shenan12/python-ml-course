import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section7_deephit import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox

st.title("🏏 Why Proportional Hazards Can Fail")
st.markdown(
    r"""
Section 6 ended on a warning. DeepSurv replaced Cox's linear predictor with a
neural network — but it left the **structure** of the model untouched:

$$h(t \mid x) = h_0(t)\cdot\exp(\hat h_\theta(x))$$

The covariates still only *scale* a shared baseline hazard. That means the
hazard ratio between any two individuals is **constant for all time**. Make the
network as deep as you like: it cannot express *"A is riskier than B early on,
but B becomes riskier later."*

**DeepHit exists because that assumption is often false.** And rather than
assert it, we're going to *test* it — on real data.

### The dataset: 25,503 real ODI innings

From [Cricsheet](https://cricsheet.org) (men's ODIs, 2010–2026, ball-by-ball).
The survival framing is beautiful, because it falls out of cricket's own rules:

| survival concept | cricket |
|---|---|
| **time** | balls faced |
| **event** | the batter is **dismissed** |
| **censoring** | the batter is **NOT OUT** — the innings ended (overs ran out, or the team won) while he was still batting |

That censoring is **real, not simulated**. We know a not-out batter survived *at
least* that many balls, but not how long he would have lasted. About **16%** of
innings end this way.
"""
)

res = artifacts.load("h1_ph_violation")
if res is None:
    st.error("Run `python build_artifacts.py --only h1` to compute this page.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("batter-innings", f"{int(res['n']):,}")
c2.metric("dismissals (events)", f"{int(res['n_events']):,}")
c3.metric("not out (censored)", f"{float(res['censor_rate']):.1%}")
st.caption(f"📦 {artifacts.provenance(res)} — computed from the real Cricsheet "
           "data by `build_artifacts.py`.")

st.header("1 · The hazard is not flat: 'playing yourself in' is real")
st.markdown(
    r"""
The **hazard** here is: *given a batter has survived $b$ balls, what's the
chance he is dismissed on ball $b+1$?* Every cricket commentator has a theory
about this. Let's just measure it, directly from 25,503 innings:
"""
)
balls = np.asarray(res["balls"])
hz = np.asarray(res["hazard"])
window = st.slider("smoothing window (balls)", 1, 15, 5)
smooth = pd.Series(hz).rolling(window, center=True, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(balls, hz, color="#b0bec5", linewidth=0.9, label="raw hazard")
ax.plot(balls, smooth, color="#c62828", linewidth=2.4,
        label=f"smoothed ({window}-ball window)")
ax.axvspan(1, 10, color="#ffcdd2", alpha=0.35)
ax.text(5.5, hz[:40].max() * 0.97, "the danger zone:\n'playing yourself in'",
        ha="center", fontsize=8, color="#c62828")
ax.set_xlabel("balls faced")
ax.set_ylabel("P(dismissed on this ball | survived so far)")
ax.set_xlim(1, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

early = hz[:10].mean()
settled = hz[30:60].mean()
st.success(
    f"""
**The folklore is true, and here is the number.** A batter's dismissal hazard
is **{early:.4f}** over his first 10 balls, falling to **{settled:.4f}** once
he's settled (balls 31–60) — he is roughly **{early / settled:.0%}** as likely
to be dismissed per ball when new at the crease. Getting "in" is worth about
{100 * (early / settled - 1):.0f}% in survival terms.

Note this alone does **not** break the Cox model — Cox is perfectly happy with
a hazard that varies over time, because $h_0(t)$ is arbitrary and unspecified.
The problem is subtler, and it's next.
"""
)

st.header("2 · The real problem: the hazard ratio *changes* over time")
st.markdown(
    """
Cox assumes the ratio between two players' hazards is a constant. So: is an
opener's hazard a fixed multiple of a lower-order batter's, at every point in
an innings? Below, we fit a **separate Cox model within each window of the
innings** and read off the opener-vs-rest hazard ratio in each.

If proportional hazards held, these four numbers would be the same.
"""
)
hr = pd.read_csv(pd.io.common.StringIO(res["hr_csv"]))
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.bar(hr["window"], hr["hazard_ratio_opener"], color="#1565c0",
       edgecolor="white")
ax.axhline(1.0, color="#37474f", linestyle="--", linewidth=1.2)
ax.text(3.4, 1.02, "no difference", fontsize=8, color="#37474f")
for i, row in hr.iterrows():
    ax.text(i, row["hazard_ratio_opener"] + 0.015,
            f"{row['hazard_ratio_opener']:.2f}", ha="center", fontsize=9)
ax.set_ylabel("hazard ratio: opener vs rest")
ax.set_title("if Cox were right, these four bars would be identical",
             fontsize=10)
ax.grid(alpha=0.25, axis="y")
st.pyplot(fig)
plt.close(fig)
st.dataframe(hr.round(4), hide_index=True, width="stretch")

st.header("3 · The formal test: Schoenfeld residuals")
st.markdown(
    r"""
As a statistician you want a **test**, not a bar chart. The standard one is
based on **Schoenfeld residuals**: for each covariate, if proportional hazards
holds, its residuals should show **no trend against time**. A significant
correlation with time = the covariate's effect is changing = **PH is violated**.

`lifelines` implements this as `proportional_hazard_test`. Here it is, run on
the real data:
"""
)
tests = pd.read_csv(pd.io.common.StringIO(res["tests_csv"]))
tests["verdict"] = np.where(
    tests["p"] < 0.05, "❌ PH VIOLATED", "✅ consistent with PH")
st.dataframe(
    tests.style.format({"test_statistic": "{:.2f}", "p": "{:.5f}"})
    .map(lambda v: "background-color: #ffcdd2"
         if isinstance(v, str) and "VIOLATED" in v else "",
         subset=["verdict"]),
    hide_index=True, width="stretch")

violated = tests[tests["p"] < 0.05]["covariate"].tolist()
st.error(
    f"""
**Proportional hazards is rejected** for **{len(violated)} of
{len(tests)}** covariates — `{'`, `'.join(violated)}` — with p-values as low as
{tests['p'].min():.1e}.

Read what that *means* in cricket terms. Take `batting_position`: an opener
faces the new ball, hard and swinging, so his hazard is elevated **early**.
Survive 20 balls and the ball is softer, the field spreads, and his hazard
drops *relative to* a middle-order batter who arrives later against a softer
ball but tighter fields and mounting run-rate pressure. **The ranking of who is
most at risk changes as the innings progresses.**

A Cox model — and therefore DeepSurv, however deep — **structurally cannot
represent this.** It is forced to report one average hazard ratio and pretend
it applies at ball 1 and ball 100 alike.
"""
)
st.info(
    f"For reference, a linear Cox model on these features scores a C-index of "
    f"**{float(res['cox_cindex']):.3f}** on this data. Keep that number in "
    "mind — everything in this section gets compared to it, and you should be "
    "sceptical of anything claiming a huge leap over it."
)

with st.expander("🎓 Deeper statistics — what a Schoenfeld residual actually "
                 "is, and how to read the test honestly"):
    st.markdown(
        r"""
The residual has a lovely construction. Recall each failure's partial-
likelihood term is a softmax over its risk set. At batter $i$'s dismissal,
the model therefore implies an *expected* covariate value for "whoever gets
dismissed here":

$$\bar x(T_i) = \sum_{j \in \mathcal{R}(T_i)} \pi_j\, x_j, \qquad \pi_j = \frac{e^{\hat\beta^\top x_j}}{\sum_{k \in \mathcal{R}(T_i)} e^{\hat\beta^\top x_k}}$$

The **Schoenfeld residual** is simply $r_i = x_i - \bar x(T_i)$ — *who
actually got out, minus who the fitted model expected to get out*, one
residual per dismissal, one component per covariate. (It's the same
observed-minus-expected template as every residual you know; here the
"expected" comes from the risk-set softmax.)

Under PH these residuals are mean-zero *at every point in time* — the
model's errors shouldn't drift as the innings progresses. So the test:
scale the residuals (by the local covariance — "scaled Schoenfeld"), plot
them against (a transform of) time, and test the slope. Grambsch and
Therneau's key result is that this slope is exactly the leading term of a
**time-varying coefficient** $\beta(t) = \beta + \theta \cdot g(t)$, so the
test is a score test of $H_0\!: \theta = 0$. Positive slope = the
covariate's effect *grows* with time.

Two honesty clauses before you cite a p-value like {tiny:.0e}:

- **With n = 25,503, significance is cheap.** A PH test's p-value measures
  *evidence that the violation is nonzero*, not that it is *large*. At this
  sample size even a trivial drift rejects. The bar chart of
  window-by-window hazard ratios above is the *effect-size* view — always
  report both, and let the ratio change (not the p-value) argue that the
  violation matters.
- **The test only sees the trend shape it looks for** (linear in $g(t)$ —
  lifelines uses rank/KM transforms of time). A perfectly symmetric
  rise-then-fall effect can slip past it, just as the U-shaped risk slipped
  past the linear Cox fit on d7. Non-rejection is "no evidence", never
  "PH confirmed".
""".replace("{tiny:.0e}", f"{tests['p'].min():.0e}")
    )

st.header("4 · Seeing it: survival curves that cross")
st.markdown(
    "The most visual symptom of non-proportional hazards is **crossing (or "
    "converging) survival curves**. Under Cox's assumption, curves may never "
    "cross — one group is uniformly riskier, forever. Here are the real "
    "Kaplan–Meier curves by batting position:"
)
fig, ax = plt.subplots(figsize=(8.5, 4))
for name, colour in [("openers (1-2)", "#1565c0"),
                     ("middle order (3-5)", "#2e7d32"),
                     ("lower order (6+)", "#c62828")]:
    t = np.asarray(res[f"km_{name}_t"])
    s = np.asarray(res[f"km_{name}_s"])
    ax.step(t, s, where="post", linewidth=2.2, color=colour, label=name)
ax.set_xlabel("balls faced")
ax.set_ylabel("P(still batting)")
ax.set_xlim(0, 90)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.set_title("Kaplan–Meier survival by batting position (real ODI data)",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Look at how the curves **change their spacing** rather than staying
proportionally apart — steep early separation, then a different relationship
once batters are set. That shifting geometry is exactly what the Schoenfeld
test just detected numerically.

### So what does DeepHit do instead?

It **throws away the hazard-ratio structure entirely.** No baseline hazard, no
proportionality, no $h_0(t)\\exp(\\cdot)$. Instead it predicts, directly, a
**probability distribution over when the event happens** — one probability per
time bin, straight out of a softmax. If the model wants a batter's risk to
peak early and fall later, it just... says so. Nothing in its structure forbids
it.

That's the next four pages: **discretise time → softmax over bins → a two-part
loss → compare.**
"""
)

guided_sandbox(
    key="h1",
    heavy=True,
    est="~15 s",
    steps="""
1. **Step 1** — load the data with `cricket.engineer(cricket.load_raw())` and
   print how many innings ended **not out** (`event == 0`). These are the
   censored observations.
2. **Step 2** — compute the empirical dismissal hazard for balls 1–40:
   for each ball `b`, `outs = ((balls_faced == b) & (event == 1)).sum()` and
   `at_risk = (balls_faced >= b).sum()`; the hazard is their ratio. Print the
   average hazard over balls 1–10 versus balls 31–40.
3. **Step 3** — fit `CoxPHFitter` on `["career_avg", "batting_position",
   "wickets_at_entry"]` with `balls_faced` / `event`, then run
   `proportional_hazard_test(cph, d, time_transform="rank")` and print the
   summary. Which covariates violate PH?
4. **Step 4 (stretch)** — split batters into openers (`batting_position <= 2`)
   and the rest, fit a `KaplanMeierFitter` to each, and plot both survival
   curves. Do they look proportionally spaced, or does their relationship
   change as the innings goes on?
""",
    setup_code='''import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from utils import cricket

df = cricket.engineer(cricket.load_raw())
print("columns:", [c for c in df.columns if not c.startswith("vs_")])
print(f"{len(df):,} innings")

# Step 1: how many innings ended NOT OUT?


# Step 2: the empirical hazard, balls 1-40


# Step 3: fit Cox, run the Schoenfeld proportional_hazard_test


# Step 4 (stretch): Kaplan-Meier curves, openers vs the rest
''',
    solution_code='''import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from utils import cricket

df = cricket.engineer(cricket.load_raw())

not_out = (df.event == 0).sum()
print(f"not out (censored): {not_out:,} of {len(df):,} "
      f"({not_out / len(df):.1%})")

hz = []
for b in range(1, 41):
    at_risk = (df.balls_faced >= b).sum()
    outs = ((df.balls_faced == b) & (df.event == 1)).sum()
    hz.append(outs / at_risk)
hz = np.array(hz)
print(f"\\nhazard, balls  1-10: {hz[:10].mean():.4f}")
print(f"hazard, balls 31-40: {hz[30:40].mean():.4f}")
print(f"-> a new batter is {hz[:10].mean() / hz[30:40].mean():.0%} as likely "
      "to be dismissed per ball. 'Playing yourself in' is real.")

feats = ["career_avg", "batting_position", "wickets_at_entry"]
d = df[feats + ["balls_faced", "event"]].dropna()
cph = CoxPHFitter().fit(d, "balls_faced", "event")
print("\\nSchoenfeld test for proportional hazards:")
zph = proportional_hazard_test(cph, d, time_transform="rank")
print(zph.summary[["test_statistic", "p"]].round(5).to_string())
print("p < 0.05 => that covariate's effect CHANGES over time => PH violated")

print("\\nKaplan-Meier by role:")
for name, mask in [("openers", df.batting_position <= 2),
                   ("rest", df.batting_position > 2)]:
    km = KaplanMeierFitter().fit(df.loc[mask, "balls_faced"],
                                 df.loc[mask, "event"])
    s = km.survival_function_.iloc[:, 0]
    at = [int(s.index.searchsorted(b)) for b in [10, 30, 60]]
    print(f"  {name:8}: S(10)={s.iloc[at[0]]:.3f}  S(30)={s.iloc[at[1]]:.3f}  "
          f"S(60)={s.iloc[at[2]]:.3f}")
print("if PH held, the RATIO of log-survivals would be constant. Check it:")''',
)
