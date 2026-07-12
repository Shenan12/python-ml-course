import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sections.section6_deepsurv import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("📊 The Paper's Results & Hyperparameter Search")

st.header("1 · What DeepSurv achieved on real data")
st.markdown(
    """
The paper evaluates on three real clinical datasets (plus a fourth pair,
Rotterdam & GBSG, used to validate the treatment recommender). **These are
genuinely medical** — that's what the authors used, and reporting them
faithfully is part of reading the paper honestly (our own simulations stay on
equipment, as you asked).

- **WHAS** — Worcester Heart Attack Study: survival after a heart attack.
- **SUPPORT** — Study to Understand Prognoses Preferences Outcomes and Risks
  of Treatment: seriously ill hospitalised patients.
- **METABRIC** — Molecular Taxonomy of Breast Cancer International
  Consortium: gene expression + clinical features.

The comparison is against **CPH** (the linear Cox model) and **RSF** (Random
Survival Forest — the survival-analysis cousin of the Random Forest you built
in Section 3: it grows trees that split to maximise survival differences).
Everything is scored by **C-index on held-out test data**.
"""
)

results = pd.DataFrame({
    "dataset": ["Simulated linear", "Simulated nonlinear", "WHAS", "SUPPORT",
                "METABRIC"],
    "n": ["6,000", "6,000", "1,638", "9,105", "1,980"],
    "events observed": ["50%", "50%", "42.1%", "68.1%", "57.7%"],
    "CPH (linear Cox)": [0.7792, 0.4867, 0.8160, 0.5831, 0.6317],
    "DeepSurv": [0.7781, 0.6524, 0.8667, 0.6189, 0.6545],
    "RSF": [0.7579, 0.6266, 0.8929, 0.6193, 0.6195],
})
st.dataframe(
    results.style.format({"CPH (linear Cox)": "{:.4f}", "DeepSurv": "{:.4f}",
                          "RSF": "{:.4f}"})
    .highlight_max(axis=1, subset=["CPH (linear Cox)", "DeepSurv", "RSF"],
                   color="#c8e6c9"),
    hide_index=True, width="stretch")

fig, ax = plt.subplots(figsize=(9.5, 3.6))
x = np.arange(len(results))
w = 0.26
for i, (col, colour) in enumerate([("CPH (linear Cox)", "#1565c0"),
                                   ("DeepSurv", "#c62828"),
                                   ("RSF", "#2e7d32")]):
    ax.bar(x + (i - 1) * w, results[col], w, label=col, color=colour,
           edgecolor="white")
ax.axhline(0.5, color="#9e9e9e", linestyle="--", linewidth=1.2)
ax.text(4.3, 0.512, "random guessing", fontsize=7.5, color="#757575")
ax.set_xticks(x)
ax.set_xticklabels(results["dataset"], fontsize=8)
ax.set_ylabel("C-index")
ax.set_ylim(0.4, 0.95)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
ax.set_title("Katzman et al. (2018), Table — C-index by dataset and model",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
**How to read this table like a statistician, not a cheerleader:**

- **DeepSurv beats CPH on all three real datasets**, but look at the *sizes*
  of the wins: WHAS +0.051, SUPPORT +0.036, METABRIC +0.023. These are real
  and consistent, but they are **modest**. The nonlinearity in real clinical
  data is evidently mild compared to the paper's deliberately brutal gaussian
  simulation (+0.166).
- **RSF beats DeepSurv on WHAS** (0.8929 vs 0.8667) — and comfortably. The
  paper does not hide this, and neither should you. There is **no free
  lunch**: a flexible tree ensemble can be the better model on a given
  dataset. DeepSurv clearly wins METABRIC; **SUPPORT is a dead heat**
  (0.6189 vs RSF's 0.6193, with overlapping confidence intervals); RSF wins
  WHAS.
- **SUPPORT's C-indices are low across the board** (0.58–0.62), even though
  it is by far the *largest* dataset. Seriously-ill hospitalised patients'
  outcomes are genuinely hard to order — a C-index near 0.6 isn't a broken
  model; it can be the honest ceiling of the data. Always ask what's
  achievable before judging a number.
- **METABRIC's margin (0.6317 → 0.6545) is the cleanest real-data win.** The
  paper reports bootstrapped confidence intervals — (0.627–0.636) vs
  (0.650–0.659) — which **don't overlap**, so this gap is real, if modest.
  That is why you report the spread and not just the point estimate — the
  cross-validation lesson from Section 3.
"""
)
st.info(
    "**Verification note.** The C-index values above are the published BMC "
    "version's Table 1 (Katzman et al. 2018, BMC Medical Research "
    "Methodology), re-checked against the full text (values rounded to 4 "
    "decimals); the *events observed* column is the paper's dataset table "
    "(so the *censoring* rates are the complements: WHAS ≈ 58%, SUPPORT "
    "≈ 32%, METABRIC ≈ 42%). Beware: the earlier **arXiv preprint reports "
    "different numbers** for the real datasets (e.g. METABRIC DeepSurv "
    "0.6434 vs the published 0.6545) — if numbers you find elsewhere "
    "disagree, check which version they came from, and cite the published "
    "one in your dissertation."
)

st.header("2 · The hyperparameter search, rebuilt and running")
st.markdown(
    """
The paper does **not** hand-pick its architecture. It runs a **random
hyperparameter search** (using the *Optunity* library with Sobol
quasi-random sampling), selecting the configuration that maximises validation
C-index under **3-fold cross-validation**. The searched hyperparameters and
the approximate ranges reported across its experiments:

| hyperparameter | range across the paper's Table 3 |
|---|---|
| hidden layers (depth) | 1 – 3 |
| nodes per layer | 4 – 48 |
| learning rate | 0.0003 – 0.154 |
| L2 (weight decay) | 2.0 – 16.1 |
| dropout | 0.11 – 0.66 |
| learning-rate decay | 0.0003 – 0.006 |
| momentum | 0.84 – 0.94 |

**Why random search, not grid search?** This is a genuinely important
methodological point, and a good viva question. With a grid, if you try 4
values of each of 6 hyperparameters, that's 4⁶ = 4,096 fits — and yet you've
still only tested **4 distinct values** of the learning rate. Random search
spends the same budget sampling *every* hyperparameter at a different value
each time, so with 50 trials you get 50 distinct learning rates. Since in
practice only a few hyperparameters really matter (and you don't know which
in advance), random search finds good configurations far more efficiently.

**Below is that search, actually running** on our nonlinear machine data —
the same loop, smaller budget:
"""
)

n_trials = st.select_slider("number of random trials", [5, 10, 20], value=10)


@st.cache_data(show_spinner="Running the random search for real — every trial "
                            "is a full DeepSurv fit…")
def random_search_live(n_trials):
    """Only ever called when you press ▶ Run the search live."""
    return experiments.random_search_experiment(n_trials)


payload = artifacts.load("d8_random_search")
live_key = "_d8_live"

if st.session_state.get(live_key):
    payload = random_search_live(n_trials)
    source_note = "just searched live on your machine"
elif payload is not None:
    # the first k trials of the saved 20-trial run ARE the trials a k-trial
    # run produces: the RNG is drawn sequentially (see _experiments.py)
    source_note = (f"{artifacts.provenance(payload)} — {experiments.D8_MAX_TRIALS} "
                   "real DeepSurv fits, saved by `build_artifacts.py`")
else:
    st.warning("No precomputed search yet. Press ▶ to run it (~1 min), or run "
               "`python build_artifacts.py` once.")
    if st.button("▶ Run the search now (~1 min)"):
        st.session_state[live_key] = True
        st.rerun()
    st.stop()

trials_df = experiments.trials_frame(payload, n_trials)

cap, btn = st.columns([3, 1])
cap.caption(
    f"📦 {source_note}. Each row below is a genuine DeepSurv model, trained "
    "with early stopping and scored on held-out data."
)
if btn.button(f"▶ Re-run {n_trials} trials live"):
    st.session_state[live_key] = True
    st.rerun()
best_i = int(trials_df["validation C-index"].idxmax())
best = trials_df.loc[best_i]

st.dataframe(
    trials_df.style.format({
        "learning rate": "{:.5f}", "dropout": "{:.2f}",
        "L2 (weight decay)": "{:.2e}", "validation C-index": "{:.4f}",
        "test C-index": "{:.4f}"})
    .highlight_max(axis=0, subset=["validation C-index"], color="#c8e6c9"),
    hide_index=True, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("best validation C-index", f"{best['validation C-index']:.4f}")
c2.metric("…its test C-index", f"{best['test C-index']:.4f}")
c3.metric("worst trial's test C-index",
          f"{trials_df['test C-index'].min():.4f}",
          delta=f"{trials_df['test C-index'].min() - best['test C-index']:+.4f}"
                " vs best")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, col, logx in [(axes[0], "learning rate", True),
                      (axes[1], "dropout", False),
                      (axes[2], "nodes", False)]:
    ax.scatter(trials_df[col], trials_df["validation C-index"], s=55,
               color="#1565c0", alpha=0.75, edgecolor="white")
    ax.scatter([best[col]], [best["validation C-index"]], s=220, marker="*",
               color="#ffd600", edgecolor="black", zorder=5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(col)
    ax.set_ylabel("validation C-index")
    ax.grid(alpha=0.25)
fig.suptitle("each dot is one real DeepSurv fit; ★ = the winner", fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    f"""
**What the search just taught us**, on real fits:

- The spread of test C-index across trials is
  **{trials_df['test C-index'].min():.3f} to
  {trials_df['test C-index'].max():.3f}**. That gap — from a badly-configured
  DeepSurv to a well-configured one — is *larger than DeepSurv's entire
  advantage over Cox on any real dataset in the paper*. **This is the
  Faraggi–Simon lesson, quantified:** an untuned neural survival model can
  easily lose to a linear Cox model, and a tuned one wins. The search isn't a
  detail of the method; in a real sense the search **is** the method.
- Look at the learning-rate panel: the relationship with performance is
  steep. It's usually the single most important knob (which is exactly why
  sampling many distinct values, as random search does, beats testing four
  grid values).
- **Selection on validation, reporting on test.** The winner was chosen by
  its *validation* C-index; we then report its *test* C-index. If you pick the
  best of 20 trials by test score, that score is optimistically biased — you
  have fitted your model *selection* to the test set. This is the burnt-test-set
  rule from Section 2, and it is the most common way otherwise-good papers
  overstate their results.
"""
)

with st.expander("🎓 Deeper statistics — the winner's curse, in order-statistic "
                 "terms"):
    st.markdown(
        r"""
Why exactly is "best of 20 by validation score" optimistic *even about the
winner itself*? Write each trial's validation C-index as
$\hat C_k = C_k + \varepsilon_k$: its true performance plus evaluation
noise (finite validation set). Selecting the maximum selects jointly on
good $C_k$ **and** lucky $\varepsilon_k$, and the expectation of a maximum
exceeds the maximum of expectations:

$$\mathbb{E}\big[\max_k (C_k + \varepsilon_k)\big] \;>\; \max_k C_k \quad\text{whenever the noise isn't degenerate.}$$

So the winner's *reported validation score* overstates its *true* skill —
the *winner's curse*, the same order-statistics fact that makes the best
fund of 20 look better than it is and the most significant of 8 noise
coefficients on the last page look "real". Consequences you can act on:

- **The winner's validation C-index is spent.** It selected the model; it
  can no longer measure it. The unbiased read is the untouched test set —
  which is why the table reports both, and why they usually disagree in the
  humble direction.
- **More trials sharpen the curse.** As $K$ grows, $\mathbb{E}[\max_k \varepsilon_k]$
  grows (roughly like $\sqrt{2\log K}\,\sigma$ for Gaussian noise), so a
  bigger search needs a *better* validation estimate to be trustworthy —
  which is exactly why the paper selects with **3-fold cross-validation**
  (averaging 3 folds shrinks $\sigma$ by $\sqrt 3$) rather than one split.
- **The same algebra polices literature reviews.** "Best method of 12 on
  benchmark X" carries the same upward bias between papers as between our
  20 trials. When a method's margin is smaller than the benchmark's
  evaluation noise, the league table is mostly ordering the
  $\varepsilon_k$'s.
"""
    )

st.header("3 · The search loop in code")
show_example(
    experiments.SNIPPETS["d8_example"],
    """
- `rng.integers(1, 4)` / `rng.choice([8, 16, 32, 48])` — depth and width sampled from the paper's ranges.
- `10 ** rng.uniform(-3.5, -1.0)` — the learning rate is sampled **log-uniformly**, i.e. uniformly across *orders of magnitude* (0.0003 to 0.1). This matters: sampling uniformly in [0.0003, 0.1] would put 90% of your trials above 0.01 and almost never test small rates. Learning rates and regularization strengths should essentially always be searched on a log scale — a small but genuinely important piece of craft.
- `tt.callbacks.EarlyStopping(patience=20)` — stop when validation loss hasn't improved for 20 epochs, and keep the best weights. This makes `epochs=200` an upper bound rather than a target, so a slow-learning configuration isn't unfairly cut off and a fast-overfitting one isn't allowed to rot.
- `if c_val > best[0]` — selection strictly on **validation**. The test set is never consulted inside this loop.
- This ~20-line loop is, in structure, exactly what the paper's appendix describes (they use Optunity for smarter quasi-random sampling and 3-fold CV, but the logic is this).
""",
    key="d8_example",
    heavy=True,
    est="~60 s",
)

guided_sandbox(
    key="d8",
    heavy=True,
    est="~90 s",
    steps="""
1. **Step 1** — write a `fit_and_score(depth, nodes, lr, dropout, wd)`
   function that builds an `MLPVanilla` + `CoxPH` model, fits it with early
   stopping on the validation data, and returns the **validation** C-index.
2. **Step 2** — run a **grid search**: 2 depths × 2 node counts × 2 learning
   rates = 8 fits. Print each result and the winner.
3. **Step 3** — run a **random search** with the same budget of 8 fits,
   sampling `lr` log-uniformly (`10 ** rng.uniform(-3.5, -1)`). Compare the
   best validation C-index against your grid search's. Which explored the
   learning rate better?
4. **Step 4 (stretch)** — take your winning configuration and evaluate it on
   the **test** set (untouched until now). Is the test C-index lower than the
   validation C-index that won? Explain to yourself why that's expected —
   and why reporting the validation number would have been dishonest.
""",
    setup_code='''import numpy as np
import torch
import torchtuples as tt
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

Xtr, Ttr, Etr = machines(n=1200, risk="nonlinear", seed=1)
Xva, Tva, Eva = machines(n=600, risk="nonlinear", seed=3)
Xte, Tte, Ete = machines(n=600, risk="nonlinear", seed=2)
mu, sd = Xtr.mean(0), Xtr.std(0)
f = lambda A: ((A - mu) / sd).astype("float32")
ytr = (Ttr.astype("float32"), Etr.astype("float32"))
yva = (Tva.astype("float32"), Eva.astype("float32"))
print("ready. NOTE: each fit takes a few seconds - keep the budget small.")

# Step 1: fit_and_score(depth, nodes, lr, dropout, wd) -> validation C-index


# Step 2: grid search over 2 depths x 2 node counts x 2 learning rates


# Step 3: random search with the same budget (log-uniform lr)


# Step 4 (stretch): the winner's TEST C-index
''',
    solution_code='''import numpy as np
import torch
import torchtuples as tt
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

Xtr, Ttr, Etr = machines(n=1200, risk="nonlinear", seed=1)
Xva, Tva, Eva = machines(n=600, risk="nonlinear", seed=3)
Xte, Tte, Ete = machines(n=600, risk="nonlinear", seed=2)
mu, sd = Xtr.mean(0), Xtr.std(0)
f = lambda A: ((A - mu) / sd).astype("float32")
ytr = (Ttr.astype("float32"), Etr.astype("float32"))
yva = (Tva.astype("float32"), Eva.astype("float32"))

def fit_and_score(depth, nodes, lr, dropout=0.2, wd=1e-4, return_model=False):
    torch.manual_seed(0)
    net = tt.practical.MLPVanilla(10, [nodes] * depth, 1, batch_norm=True,
                                  dropout=dropout, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(lr, weight_decay=wd))
    model.fit(f(Xtr), ytr, batch_size=256, epochs=150, verbose=False,
              val_data=(f(Xva), yva),
              callbacks=[tt.callbacks.EarlyStopping(patience=15)])
    c = concordance_index(Tva, -model.predict(f(Xva)).ravel(), Eva)
    return (c, model) if return_model else c

print("--- GRID search (8 fits) ---")
grid_best = None
for depth in [1, 2]:
    for nodes in [16, 32]:
        for lr in [0.001, 0.01]:
            c = fit_and_score(depth, nodes, lr)
            print(f"depth={depth} nodes={nodes} lr={lr:<6}: val {c:.4f}")
            if grid_best is None or c > grid_best[0]:
                grid_best = (c, depth, nodes, lr)
print(f"grid winner: val C-index {grid_best[0]:.4f}")
print("note: the grid only ever tried TWO distinct learning rates.")

print("\\n--- RANDOM search (8 fits, same budget) ---")
rng = np.random.default_rng(7)
rand_best = None
for t in range(8):
    depth = int(rng.integers(1, 4))
    nodes = int(rng.choice([8, 16, 32, 48]))
    lr = float(10 ** rng.uniform(-3.5, -1.0))
    drop = float(rng.uniform(0.1, 0.6))
    c = fit_and_score(depth, nodes, lr, drop)
    print(f"depth={depth} nodes={nodes} lr={lr:.5f} drop={drop:.2f}: "
          f"val {c:.4f}")
    if rand_best is None or c > rand_best[0]:
        rand_best = (c, depth, nodes, lr, drop)
print(f"random winner: val C-index {rand_best[0]:.4f} "
      f"(8 DISTINCT learning rates tried)")

c_val, model = fit_and_score(rand_best[1], rand_best[2], rand_best[3],
                             rand_best[4], return_model=True)
c_test = concordance_index(Tte, -model.predict(f(Xte)).ravel(), Ete)
print(f"\\nwinner's validation C-index: {c_val:.4f}")
print(f"winner's TEST C-index:       {c_test:.4f}")
print("The test score is usually a little lower: we CHOSE this config "
      "because it scored well on validation, so that score is flattering. "
      "The test set is the only unbiased estimate - report it.")''',
)
