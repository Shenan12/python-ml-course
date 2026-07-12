import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section7_deephit import _experiments as experiments
from utils import artifacts, cricket
from utils.sandbox import guided_sandbox, show_example

st.title("🛠️ The Cricket Data & Feature Engineering")
st.markdown(
    """
Before any model, the data. This page is the one that most resembles a real
job: **the modelling is 20% of the work and the features are the other 80%**.
Everything here is done on the actual Cricsheet ball-by-ball file — you can
read the whole pipeline in
[`scripts/prepare_cricket_data.py`](scripts/prepare_cricket_data.py).

### From 1.35 million deliveries to 25,503 innings

The raw file is one row per **ball**. We need one row per **batter-innings**,
with a duration and an event. Three decisions had to be made, and each is the
kind of thing an examiner will poke at:
"""
)

st.markdown(
    """
| decision | what we did | why |
|---|---|---|
| **What counts as a "ball faced"?** | count every delivery except **wides** | a wide is not a ball faced by the batter — cricket's own scoring rule. Getting this wrong would corrupt the duration for every innings. |
| **What is the event?** | the batter appears in `player_dismissed` | note a batter can be **run out at the non-striker's end** — so we can't just look at who was on strike. We search the whole innings for his dismissal. |
| **What about "retired hurt"?** | treat as **censored**, not dismissed | he left without being out. Counting it as a dismissal would be recording an event that never happened. |
"""
)

res = artifacts.load("h2_leakage")
if res is None:
    st.error("Run `python build_artifacts.py --only h2` to compute this page.")
    st.stop()

st.header("1 · The censoring is real")
try:
    df = cricket.engineer(cricket.load_raw())
except FileNotFoundError:
    st.error(
        "**The cricket dataset is missing** (`data/odi_batting_innings.csv`). "
        "If you are deploying, make sure the `data/` folder is committed to "
        "the repository. To rebuild it locally: "
        "`python scripts/prepare_cricket_data.py`."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("innings", f"{len(df):,}")
c2.metric("dismissed", f"{int(df.event.sum()):,}")
c3.metric("NOT OUT (censored)", f"{int((1 - df.event).sum()):,}",
          f"{1 - df.event.mean():.1%}")
c4.metric("median balls faced", f"{int(df.balls_faced.median())}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].hist([df.loc[df.event == 1, "balls_faced"],
              df.loc[df.event == 0, "balls_faced"]],
             bins=40, stacked=True, color=["#c62828", "#2e7d32"],
             label=["dismissed (event)", "not out (censored)"])
axes[0].set_xlabel("balls faced")
axes[0].set_ylabel("innings")
axes[0].legend(fontsize=8)
axes[0].set_title("the duration distribution", fontsize=10)
axes[0].grid(alpha=0.2)

cens_by_pos = df.groupby("batting_position")["event"].apply(
    lambda s: 1 - s.mean())
cens_by_pos = cens_by_pos[cens_by_pos.index <= 11]
axes[1].bar(cens_by_pos.index, cens_by_pos.values, color="#2e7d32",
            edgecolor="white")
axes[1].set_xlabel("batting position")
axes[1].set_ylabel("proportion NOT OUT")
axes[1].set_title("censoring is not random!", fontsize=10)
axes[1].grid(alpha=0.2, axis="y")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.warning(
    f"""
**Look hard at the right-hand chart — this is a genuinely important subtlety.**

Censoring is **not uniform**: tail-enders are not out far more often than
openers ({cens_by_pos.iloc[-1]:.0%} vs {cens_by_pos.iloc[0]:.0%}). Why?
Because the innings ends *around them* — the overs run out, or the team wins,
and the No. 10 is stranded on 4 not out.

Standard survival analysis assumes **non-informative censoring**: that being
censored tells you nothing about your risk beyond what the covariates already
say. Here, censoring is clearly *related* to batting position — but batting
position **is one of our covariates**, so the assumption is one of *conditional*
independence, which is more defensible. Still, this is exactly the kind of
assumption you should state explicitly in a dissertation rather than assume
silently. It is a limitation of this dataset, and an honest one to name.
"""
)

with st.expander("🎓 Deeper statistics — writing the censoring assumption "
                 "down properly"):
    st.markdown(
        r"""
Let $T$ be the (possibly unobserved) balls-until-dismissal and $C$ the
balls-until-the-innings-ends-around-him. We observe
$Y = \min(T, C)$ and $E = \mathbb{1}[T \le C]$ — the classic
**random censorship model**. The assumption each estimator needs, from
weakest to strongest:

- **Kaplan–Meier (marginal curves):** $T \perp C$ *unconditionally*. The
  right-hand chart above shows this is **false** here — censoring
  probability varies sharply with batting position, and position also
  predicts $T$.
- **Cox / DeepSurv / DeepHit (covariate models):** only
  $T \perp C \mid X$ — *conditionally* independent censoring. Given the
  covariates (position, entry over, match situation), the residual
  randomness in "does the innings end around him" must carry no extra
  information about how long he'd have batted. Because position and entry
  situation are in $X$, this is far more defensible — the *observable*
  driver of the dependence is conditioned away.
- What would still break it: something *unobserved* that drives both — e.g.
  batters who accelerate when the end is near change their dismissal risk
  *because* $C$ is close. That residual dependence is untestable from this
  data alone (censored innings never reveal their $T$), which is why the
  page says "state it, don't assume it silently".

One more connection worth a line in a dissertation: "innings ends" is
really a **competing risk** for "dismissed" — the not-out ending removes
the batter from observation just as death-from-other-causes removes a
patient. Treating a competing event as censoring is exactly the situation
DeepHit's full multi-cause machinery (introduced on the architecture page)
was designed for; we use the single-risk version because for *this*
question — dismissal risk while batting — the censoring treatment is the
standard and defensible choice.
"""
    )

st.header("2 · The features, and why each one exists")
st.markdown(
    """
Every feature had to be **available before the first ball of the innings** —
otherwise we'd be predicting the past from the future.
"""
)
st.markdown(
    """
| feature | what it captures |
|---|---|
| `career_avg` | runs per dismissal **before this innings** — class |
| `career_sr` | strike rate before this innings — style (aggressive batters take more risk) |
| `career_balls_per_dismissal` | *survival-native* form: how long does he usually last? |
| `career_innings` | experience — a debutant is a different animal from a 200-cap veteran |
| `batting_position` | 1 = opener, facing the new ball |
| `entry_over` | when he walked in |
| `runs_at_entry`, `wickets_at_entry` | the match situation. Arriving at 15/3 ≠ arriving at 150/1 |
| `run_rate_at_entry` | **engineered**: runs ÷ overs. The *rate* matters more than the raw score — 150 in 20 overs is a platform, 150 in 45 is a crisis |
| `is_chasing` | 2nd innings: scoreboard pressure is a different game |
| `vs_India`, `vs_Australia`, … | the bowling attack, **one-hot encoded** (Section 3: never encode nominal categories as ordered integers) |
"""
)

show_example(
    experiments.SNIPPETS["h2_features"],
    """
- `cricket.engineer(cricket.load_raw())` — loads the prepared CSV and adds the derived features (one-hot opposition, run rate at entry).
- The printed table is the actual modelling frame: one row per batter-innings, with `balls_faced` as the duration and `event` as the dismissal indicator.
- `cricket.splits(df)` — the **time-ordered** split (see below). Notice the year ranges don't overlap: we train on the past and test on the future, exactly as you'd have to in reality.
- `feature_columns` — the final feature list handed to the models. Roughly 20 columns, most of them one-hot opposition flags.
""",
    key="h2_features",
    heavy=True,
    est="~5 s",
)

st.header("3 · The leakage trap — and how big it actually is")
st.markdown(
    r"""
Two forms of leakage stalk this dataset, and the second one nearly caught me.

**Trap 1: career statistics.** The obvious feature is "his career average". But
a batter's *final* career average includes the very innings you're predicting —
and every innings after it. Using it would let the model peek at the future. The
fix is an **expanding, shifted** statistic: for each innings, compute his
average using **only innings played strictly before it**. That's what
`add_career_features` does, with `cumsum() - current_value`.

**Trap 2: the split itself.** Even with properly lagged features, splitting the
innings *at random* means the model trains on 2024 matches and is tested on 2018
ones. It can learn "this era scores faster", and a batter's own past and future
innings land on both sides of the split. The honest choice is a **time-ordered
split**:
"""
)
c1, c2, c3 = st.columns(3)
c1.metric("train", f"{int(res['n_train']):,}", res["train_years"])
c2.metric("validation", f"{int(res['n_val']):,}", "2020–2022")
c3.metric("test", f"{int(res['n_test']):,}", res["test_years"])

c_time = float(res["c_time_split"])
c_random = float(res["c_random_split"])
fig, ax = plt.subplots(figsize=(7, 2.4))
ax.barh(["time-ordered split\n(honest)", "random split\n(leaky)"],
        [c_time, c_random], color=["#2e7d32", "#c62828"], edgecolor="white")
for i, v in enumerate([c_time, c_random]):
    ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=10)
ax.set_xlim(0.55, max(c_time, c_random) + 0.02)
ax.set_xlabel("C-index (DeepSurv, identical model & features)")
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(
    f"""
**Report the truth, including when it's undramatic.** The random split scores
**{c_random:.4f}**; the honest time-ordered split scores **{c_time:.4f}**. The
leak is worth **+{c_random - c_time:.4f}** — real, in the direction theory
predicts, but *small*.

Why so small? **Because we already defused the big trap.** The career features
are lagged by construction, so the random split can only leak era effects and
cross-innings correlation — the crumbs. Had we used each batter's *full-career*
average, the gap would be far uglier.

The lesson is the one that matters for your dissertation: **the fix that
mattered was invisible in the final number.** If I had used full-career averages
and a random split, I'd have reported a flattering C-index and never known. You
cannot detect this kind of leakage by looking at your results — only by
reasoning about how each feature was constructed.
"""
)

guided_sandbox(
    key="h2",
    heavy=True,
    est="~10 s",
    steps="""
1. **Step 1** — for one famous batter (try `"V Kohli"` or `"JE Root"`), pull
   his innings sorted by date and print `career_avg` alongside `runs` for his
   first 10 innings in the data. Confirm `career_avg` only ever reflects
   **earlier** innings.
2. **Step 2** — build the *leaky* version to see the difference: compute each
   batter's **full-career** average (a plain `groupby('batter')['runs'].mean()`
   over the whole dataset) and merge it back. Print both columns side by side
   for your batter. Which one "knows the future"?
3. **Step 3** — print the censoring rate (`1 - event.mean()`) by
   `batting_position`. Explain to yourself why tailenders are so often not out,
   and why that matters for the non-informative-censoring assumption.
4. **Step 4 (stretch)** — engineer a new feature of your own and check whether
   it is legal (available before the first ball). Ideas: `balls_remaining` in
   the innings at entry, `is_home` (batting_team in the venue's country), or
   `chasing_target_pressure`. Which of your ideas would leak?
""",
    setup_code='''import numpy as np
import pandas as pd
from utils import cricket

df = cricket.engineer(cricket.load_raw())
print(f"{len(df):,} innings, {df.batter.nunique()} batters")
print("top batters by innings:",
      df.batter.value_counts().head(5).index.tolist())

# Step 1: one batter's career_avg over his first 10 innings


# Step 2: build the LEAKY full-career average and compare


# Step 3: censoring rate by batting position


# Step 4 (stretch): invent a feature - is it available before ball 1?
''',
    solution_code='''import numpy as np
import pandas as pd
from utils import cricket

df = cricket.engineer(cricket.load_raw())

name = df.batter.value_counts().index[0]      # the most-featured batter
one = df[df.batter == name].sort_values("start_date")
print(f"--- {name}: first 10 innings in the data ---")
print(one[["start_date", "runs", "balls_faced", "event",
           "career_avg"]].head(10).to_string(index=False))
print("note career_avg on row k uses ONLY rows before k. No peeking.")

leaky = df.groupby("batter")["runs"].mean().rename("leaky_full_career_avg")
one2 = one.merge(leaky, on="batter", how="left")
print(f"\\nlagged career_avg (first innings shown): "
      f"{one2.career_avg.iloc[0]:.2f}")
print(f"LEAKY full-career avg (same row):        "
      f"{one2.leaky_full_career_avg.iloc[0]:.2f}  <- includes his FUTURE")

print("\\ncensoring (not out) rate by batting position:")
cens = df[df.batting_position <= 11].groupby(
    "batting_position")["event"].apply(lambda s: 1 - s.mean())
for pos, rate in cens.items():
    print(f"  position {int(pos):2d}: {rate:5.1%} not out")
print("tailenders are stranded when the innings ends -> censoring depends on")
print("position -> we must include position as a covariate (we do).")

df["balls_left_at_entry"] = 300 - df["entry_over"] * 6
print("\\nnew feature `balls_left_at_entry`: LEGAL (known when he walks in)")
print(df[["entry_over", "balls_left_at_entry"]].head(3).to_string(index=False))
print("\\nIllegal example: `runs_scored_this_innings` - that IS the outcome.")''',
)
