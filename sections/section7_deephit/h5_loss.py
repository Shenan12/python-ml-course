import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("⚖️ The Two-Part Loss")
st.markdown(
    r"""
DeepHit's loss is the part people quote and rarely explain. It has **two terms
added together**, and they are responsible for two genuinely different things:

$$\mathcal{L} = \underbrace{\alpha \cdot \mathcal{L}_1}_{\text{be right}} \;+\; \underbrace{(1-\alpha) \cdot \mathcal{L}_2}_{\text{be in the right order}}$$

- $\mathcal{L}_1$ — **the likelihood term.** Did you put probability mass in the
  right place? It makes the model *calibrated*: its curves should match reality.
- $\mathcal{L}_2$ — **the ranking term.** Did you assign a higher risk to the
  batter who actually got out first? It makes the model *discriminative*: good
  at ordering people.

$\alpha$ is the dial between them. This is unusual and worth pausing on: **most
losses optimise likelihood alone and hope ranking follows.** DeepHit optimises
the evaluation metric (concordance) *directly*, as part of its objective.
"""
)

st.header("1 · The likelihood term ℒ₁ — and how censoring enters")
st.markdown(
    r"""
For an **uncensored** batter (dismissed in bin $k^*$), we want the probability
mass in that bin to be high. Standard cross-entropy — exactly what you'd write
for a classifier:

$$\mathcal{L}_1^{\text{uncensored}} = -\log\big(p_{k^*}(x)\big)$$

For a **censored** batter (**not out** after bin $k^*$), we don't know which bin
he'd have been dismissed in — only that it's **later than $k^*$**. So we reward
the model for putting mass *anywhere beyond* $k^*$, i.e. for a high survival
probability at that point:

$$\mathcal{L}_1^{\text{censored}} = -\log\Big(\underbrace{\textstyle\sum_{j > k^*} p_j(x)}_{= \,S(k^*)}\Big)$$

**This is where the not-out batters do their work.** They can't say "he was
dismissed here", but they can say *"whatever happens, it wasn't by ball 40"* —
and that constrains the model. Sixteen percent of our innings contribute
exactly this way.
"""
)

st.header("2 · The ranking term ℒ₂ — optimising the C-index directly")
st.markdown(
    r"""
Take a **comparable pair** $(i, j)$: batter $i$ was dismissed at $k_i$, and
batter $j$ was still batting then. The model gets it right if it predicted $i$
was more likely to be out by that time — i.e. if $F_i(k_i) > F_j(k_i)$ (recall
$F$ = the CIF from the last page).

DeepHit penalises the pairs it gets **wrong**, with a smooth, differentiable
penalty:

$$\mathcal{L}_2 = \sum_{\text{comparable } (i,j)} \eta\Big(F_i(k_i) - F_j(k_i)\Big), \qquad \eta(z) = \exp\!\left(\frac{-z}{\sigma}\right)$$

Look at what $\eta$ does. If the model got the pair **right**, $z > 0$, so
$\exp(-z/\sigma)$ is small — little penalty. If it got the pair **wrong**,
$z<0$, and the penalty grows *exponentially*. The concordance rule ("did you
rank them correctly?") is a **step function** and has no usable gradient; this
exponential is its smooth, trainable stand-in.

$\sigma$ controls how sharp that stand-in is: small $\sigma$ ≈ a hard step
(strong gradients, but only very near the boundary); large $\sigma$ = a gentle
slope (softer, more forgiving gradients).
"""
)

sigma = st.select_slider("σ — the smoothness of the ranking penalty",
                         [0.05, 0.1, 0.5, 1.0], value=0.5)
z = np.linspace(-1, 1, 400)
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.plot(z, np.exp(-z / sigma), color="#6a1b9a", linewidth=2.4,
        label=f"η(z) = exp(−z/σ),  σ={sigma}")
ax.plot(z, (z < 0).astype(float), color="#37474f", linestyle="--",
        linewidth=1.6, label="the TRUE rule (a step — no gradient!)")
ax.axvline(0, color="#b0bec5", linewidth=1)
ax.set_xlabel("z = F_i(k_i) − F_j(k_i)      (positive = correctly ranked)")
ax.set_ylabel("penalty")
ax.set_ylim(0, 4)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.text(-0.75, 3.4, "WRONG ORDER\n(penalty explodes)", fontsize=8,
        color="#c62828")
ax.text(0.35, 0.35, "right order\n(penalty ≈ 0)", fontsize=8, color="#2e7d32")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(
    "The dashed step is what we actually care about; it has zero gradient "
    "everywhere and is useless for training. The purple curve is the "
    "differentiable stand-in DeepHit optimises instead — a recurring trick in "
    "deep learning whenever the true objective isn't differentiable."
)

st.header("3 · What α actually buys you — measured")
st.markdown(
    "Five real DeepHitSingle models on the cricket data, identical except for "
    "α. Watch the two metrics pull in different directions:"
)
res = artifacts.load("h5_alpha")
if res is None:
    st.error("Run `python build_artifacts.py --only h5`.")
    st.stop()
alpha_df = pd.read_csv(io.StringIO(res["results_csv"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(alpha_df["alpha"], alpha_df["C-td"], marker="o", color="#1565c0",
             linewidth=2.2)
axes[0].set_xlabel("α  (0 = pure ranking → 1 = pure likelihood)")
axes[0].set_ylabel("C-td (discrimination)")
axes[0].set_title("higher is better", fontsize=10)
axes[1].plot(alpha_df["alpha"], alpha_df["IBS"], marker="s", color="#c62828",
             linewidth=2.2)
axes[1].set_xlabel("α")
axes[1].set_ylabel("IBS (calibration error)")
axes[1].set_title("lower is better", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(
    alpha_df.style.format({"alpha": "{:.1f}", "C-td": "{:.4f}",
                           "IBS": "{:.4f}"})
    .highlight_max(subset=["C-td"], color="#c8e6c9")
    .highlight_min(subset=["IBS"], color="#c8e6c9"),
    hide_index=True, width="stretch")
st.caption(f"📦 {artifacts.provenance(res)} — five real models, trained on the "
           "cricket data.")

best_c = alpha_df.loc[alpha_df["C-td"].idxmax()]
best_i = alpha_df.loc[alpha_df["IBS"].idxmin()]
st.info(
    f"""
**The two metrics do not agree, and that is the entire point of having α.**

Best **discrimination** (C-td = {best_c['C-td']:.4f}) at α = **{best_c['alpha']:.1f}**;
best **calibration** (IBS = {best_i['IBS']:.4f}) at α = **{best_i['alpha']:.1f}**.

- **α = 0** is *pure ranking*. The model is told only "get the order right" and
  is never asked whether its probabilities mean anything. It can rank people
  beautifully while its survival curves are nonsense.
- **α = 1** is *pure likelihood* — this is essentially a discrete-time survival
  model with no ranking term, and it optimises calibration alone.
- **In between** you buy some of each.

**Which should you pick?** It depends on the decision the model serves, and this
is a question for you, not for the loss function:

- *"Which three batters are most likely to be out in the next 10 balls?"* — a
  **ranking** question. Favour low α.
- *"What is the probability this batter survives to face 50 balls?"* — a
  **calibration** question, and the number will be quoted as a real
  probability. Favour high α.

This connects straight back to Section 3: the C-index measures **discrimination**
(ordering), the Brier score measures **calibration** (are the probabilities
truthful?). A model can be superb at one and terrible at the other. Report both.
"""
)

with st.expander("🎓 Deeper statistics — ℒ₁ is an old friend, ℒ₂ is a smooth "
                 "U-statistic, and their sum is *not* a likelihood"):
    st.markdown(
        r"""
**ℒ₁ you have already met — twice.** On the first survival page we wrote
the censored-data likelihood $L = \prod_i f(T_i)^{E_i} S(T_i)^{1-E_i}$.
Replace density with PMF ($f \to p_{k^*}$) and take the negative log, and
you get *exactly* $\mathcal{L}_1$: $-\log p_{k^*}$ for the dismissed,
$-\log S(k^*)$ for the not-out. So $\mathcal{L}_1$ is the discrete-time
censored MLE objective — nothing invented, just the d1 likelihood with
bins. (And for uncensored data it's categorical cross-entropy, closing the
loop with the multinomial-GLM reading from the previous page.)

**ℒ₂ is a smoothed C-index.** The C-index (d1) is a U-statistic: an
average of the indicator $\mathbb{1}[\text{wrong order}]$ over comparable
pairs. Indicators have zero gradient, so DeepHit substitutes the
exponential $\eta(z) = e^{-z/\sigma}$ — an upper bound on the indicator
that *is* differentiable. This is a standard move with a family history:
logistic regression's log-loss and the SVM's hinge loss are the same trick
applied to the 0-1 classification error, and RankBoost/RankNet apply it to
ranking exactly as here. $\sigma$ is the **temperature**: as
$\sigma \to 0$, $\eta$ approaches the true step (faithful, but gradients
vanish except at the boundary — the sigmoid-saturation problem in new
clothes); large $\sigma$ gives smooth gradients that only loosely track
concordance. That's a bias-of-the-surrogate vs trainability trade, and
it's why σ needs tuning at all.

**And the sum is deliberately *not* a likelihood.** The moment
$\alpha < 1$, the objective stops being the log of any probability model —
it's a **penalised M-estimator**, likelihood plus a ranking penalty. Two
consequences you can *see in the charts above*:

1. The fitted PMFs are no longer the MLE of any distribution, so their
   probabilities drift off-calibration as $\alpha$ falls — that's the IBS
   panel rising as α → 0. In proper-scoring-rule language: log-loss is a
   proper score (truth-telling is optimal), $\eta$ is not, and mixing them
   trades honesty for order.
2. Classical likelihood-based inference (standard errors from the Hessian,
   likelihood-ratio tests, AIC) loses its justification for this objective.
   You evaluate DeepHit by held-out metrics, not information criteria —
   which is exactly what this section does.
"""
    )

st.header("4 · The loss, coded from scratch")
show_example(
    '''import numpy as np

# One batter, 6 time bins. The network's PMF for him:
pmf = np.array([0.30, 0.25, 0.18, 0.12, 0.09, 0.06])
surv = 1 - np.cumsum(pmf)

# --- L1, the likelihood term ---
# Case A: he was DISMISSED in bin 2
k_star = 2
loss_uncensored = -np.log(pmf[k_star])
print(f"dismissed in bin {k_star}: -log(pmf[{k_star}]) = "
      f"-log({pmf[k_star]:.2f}) = {loss_uncensored:.4f}")

# Case B: he was NOT OUT after bin 2 (censored)
loss_censored = -np.log(surv[k_star])
print(f"NOT OUT after bin {k_star}: -log(S({k_star})) = "
      f"-log({surv[k_star]:.2f}) = {loss_censored:.4f}")
print("  -> the censored batter only says 'not by here', so we reward")
print("     probability mass placed ANYWHERE later. He still teaches us.")
print()

# --- L2, the ranking term ---
# batter i was dismissed at bin 2; batter j was still batting then.
cif_i = np.cumsum(np.array([0.30, 0.25, 0.18, 0.12, 0.09, 0.06]))
cif_j = np.cumsum(np.array([0.05, 0.08, 0.12, 0.20, 0.25, 0.30]))
sigma = 0.5

z = cif_i[k_star] - cif_j[k_star]     # want this POSITIVE (i riskier than j)
penalty = np.exp(-z / sigma)
print(f"F_i(2) = {cif_i[k_star]:.2f}  (the one who actually got out)")
print(f"F_j(2) = {cif_j[k_star]:.2f}  (the one still batting)")
print(f"z = {z:+.2f}  -> ranked CORRECTLY" if z > 0 else f"z = {z:+.2f} -> WRONG")
print(f"ranking penalty exp(-z/sigma) = {penalty:.4f}")

# now flip it: the model got the pair backwards
z_bad = cif_j[k_star] - cif_i[k_star]
print(f"\\nif the model had it BACKWARDS: z = {z_bad:+.2f}, "
      f"penalty = {np.exp(-z_bad / sigma):.4f}  <- much bigger")

alpha = 0.5
total = alpha * loss_uncensored + (1 - alpha) * penalty
print(f"\\ntotal loss (alpha={alpha}): {alpha}*{loss_uncensored:.3f} + "
      f"{1-alpha}*{penalty:.3f} = {total:.4f}")''',
    """
- `-np.log(pmf[k_star])` — cross-entropy for the dismissed batter: the loss falls as the model puts more mass in the bin where he actually went. Identical in form to the classification loss from Section 5.
- `-np.log(surv[k_star])` — the censored batter's term. `surv[k_star]` is the total probability mass *beyond* his last observed bin, so this rewards the model for saying "he'd have lasted longer". **A not-out batter is not a missing data point; he is a one-sided constraint.**
- `z = cif_i[k_star] - cif_j[k_star]` — the comparison at the moment `i` was dismissed. We want the model to have given `i` the higher cumulative risk *at that time*.
- `np.exp(-z / sigma)` — the smooth penalty. Notice how much larger it becomes when the pair is ranked backwards: that asymmetry is what drags the model toward correct orderings.
- The final line — the two losses simply added with weights α and 1−α. That's the whole objective; nothing else is hiding in it.
""",
)

guided_sandbox(
    key="h5",
    steps="""
1. **Step 1** — write `softmax(z)` and use it to turn the `logits` in the
   editor into a PMF, then compute the survival curve.
2. **Step 2** — write `l1(pmf, k_star, event)`: return `-log(pmf[k_star])` if
   `event == 1` (dismissed), else `-log(S(k_star))` (not out). Print the loss
   for the same batter under both assumptions and explain the difference.
3. **Step 3** — write `l2(cif_i, cif_j, k_star, sigma)` returning
   `exp(-(cif_i[k_star] - cif_j[k_star]) / sigma)`. Evaluate it for a correctly
   ranked pair and a wrongly ranked pair.
4. **Step 4 (stretch)** — sweep `sigma` over `[0.05, 0.1, 0.5, 1.0]` for a
   *wrongly* ranked pair and print the penalty each time. What does a very
   small sigma do to the gradient the model receives — and what might that do
   to training stability?
""",
    setup_code='''import numpy as np

logits = np.array([1.8, 1.5, 0.9, 0.4, -0.2, -0.8])
logits_j = np.array([-1.0, -0.5, 0.2, 0.8, 1.2, 1.5])   # a safer batter
k_star = 2      # the bin in which batter i's innings ended

# Step 1: softmax -> pmf -> survival


# Step 2: l1(pmf, k_star, event) for event=1 and event=0


# Step 3: l2(cif_i, cif_j, k_star, sigma) - right pair vs wrong pair


# Step 4 (stretch): sweep sigma on a wrongly-ranked pair
''',
    solution_code='''import numpy as np

logits = np.array([1.8, 1.5, 0.9, 0.4, -0.2, -0.8])
logits_j = np.array([-1.0, -0.5, 0.2, 0.8, 1.2, 1.5])
k_star = 2

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

pmf = softmax(logits)
surv = 1 - np.cumsum(pmf)
print("PMF: ", pmf.round(3))
print("S(t):", surv.round(3))

def l1(pmf, k_star, event):
    if event == 1:
        return -np.log(pmf[k_star])
    surv = 1 - np.cumsum(pmf)
    return -np.log(max(surv[k_star], 1e-12))

print(f"\\ndismissed in bin {k_star}: L1 = {l1(pmf, k_star, 1):.4f}")
print(f"NOT OUT after bin {k_star}: L1 = {l1(pmf, k_star, 0):.4f}")
print("the censored batter constrains the model differently - he says")
print("'not by here', rewarding mass placed LATER.")

cif_i = np.cumsum(pmf)
cif_j = np.cumsum(softmax(logits_j))

def l2(cif_i, cif_j, k_star, sigma):
    z = cif_i[k_star] - cif_j[k_star]
    return np.exp(-z / sigma), z

pen_right, z_right = l2(cif_i, cif_j, k_star, 0.5)
pen_wrong, z_wrong = l2(cif_j, cif_i, k_star, 0.5)
print(f"\\ncorrectly ranked: z={z_right:+.3f}  penalty={pen_right:.4f}")
print(f"wrongly  ranked: z={z_wrong:+.3f}  penalty={pen_wrong:.4f}")

print("\\nsigma sweep on the WRONGLY ranked pair:")
for s in [0.05, 0.1, 0.5, 1.0]:
    p, _ = l2(cif_j, cif_i, k_star, s)
    print(f"  sigma={s:<5}: penalty = {p:12.2f}")
print("tiny sigma -> astronomically large penalties -> huge, unstable")
print("gradients. This is why sigma is a hyperparameter worth tuning, and")
print("why gradient clipping (Section 6) earns its keep here.")''',
)
