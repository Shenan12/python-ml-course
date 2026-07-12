import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🎯 Softmax → a Probability Mass Function over Bins")
st.markdown(
    r"""
Time is now $K$ bins. The network outputs $K$ raw numbers (logits) — and a
**softmax** turns them into a probability distribution:

$$P(\text{dismissed in bin } k \mid x)
= \frac{e^{o_k(x)}}{\sum_{j=1}^{K} e^{o_j(x)}}$$

You met softmax on the activations page as "the multi-class classifier's output
layer". Here it is doing exactly the same job — the "classes" are just *time
bins*. The outputs are non-negative and sum to 1, so they form a genuine
**probability mass function (PMF)** over when the batter gets out.

**This is the whole mechanical difference from DeepSurv.** Compare:

| | DeepSurv | DeepHit |
|---|---|---|
| network outputs | **1** number | **K** numbers |
| final activation | none (a raw log-risk) | **softmax** |
| what it means | a risk *score* (unitless, only rankable) | $P(\text{event in bin } k)$ — an actual probability |
| survival curve | reconstructed *afterwards* via a separately-estimated baseline hazard $h_0(t)$ | read straight off the PMF |
| hazard shape | forced: $h_0(t)\exp(\hat h(x))$ | free: any shape the softmax can express |

DeepSurv gives you a number and makes you assemble the curve later. **DeepHit
gives you the curve.**
"""
)

res = artifacts.load("h4_pmf")
if res is None:
    st.error("Run `python build_artifacts.py --only h4` to compute this page.")
    st.stop()

st.header("1 · From PMF to survival curve — three quantities, one object")
st.markdown(
    r"""
Once you have the PMF, everything else is arithmetic — no model needed:

- **PMF**: $\;p_k = P(\text{out in bin } k)$ — the softmax output itself.
- **CIF** (cumulative incidence): $\;F(k) = \sum_{j \le k} p_j$ — "the chance
  he's out **by** bin $k$". A running total.
- **Survival**: $\;S(k) = 1 - F(k)$ — "the chance he's **still batting** after
  bin $k$".

That's it. $S = 1 - \text{cumsum}(\text{softmax})$. The survival curve is a
*derived quantity*, and you never estimate a baseline hazard at all.
"""
)

meta = pd.read_csv(io.StringIO(res["meta_csv"]))
labels = meta["label"].tolist()
choice = st.selectbox("pick a real batter from the test set", labels)
row = meta[meta.label == choice].iloc[0]
key = choice.replace(" ", "_").replace(",", "")

pmf = np.asarray(res[f"pmf_{key}"])
surv = np.asarray(res[f"surv_{key}"])
cuts = np.asarray(res["cuts"])
surv_index = np.asarray(res["surv_index"])
cif = np.cumsum(pmf)

c1, c2, c3, c4 = st.columns(4)
c1.metric("batter", str(row["batter"]))
c2.metric("position", int(row["position"]))
c3.metric("career avg (before)", f"{float(row['career_avg']):.1f}")
c4.metric("what actually happened",
          f"{int(row['balls_faced'])} balls",
          "dismissed" if int(row["event"]) == 1 else "NOT OUT (censored)",
          delta_color="off")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
axes[0].bar(range(len(pmf)), pmf, color="#7e57c2", edgecolor="white")
axes[0].set_xlabel("time bin")
axes[0].set_ylabel("P(dismissed in this bin)")
axes[0].set_title("PMF — the softmax output", fontsize=10)

axes[1].step(range(len(cif)), cif, where="post", color="#e65100", linewidth=2.2)
axes[1].set_xlabel("time bin")
axes[1].set_ylabel("F(k)")
axes[1].set_ylim(0, 1.02)
axes[1].set_title("CIF — cumulative sum of the PMF", fontsize=10)

axes[2].step(surv_index, surv, where="post", color="#1565c0", linewidth=2.4)
if int(row["event"]) == 1:
    axes[2].axvline(float(row["balls_faced"]), color="#c62828",
                    linestyle="--", linewidth=1.6,
                    label=f"actually out on ball {int(row['balls_faced'])}")
else:
    axes[2].axvline(float(row["balls_faced"]), color="#2e7d32",
                    linestyle="--", linewidth=1.6,
                    label=f"not out on {int(row['balls_faced'])}")
axes[2].set_xlabel("balls faced")
axes[2].set_ylabel("S(t) = P(still batting)")
axes[2].set_ylim(0, 1.02)
axes[2].legend(fontsize=8)
axes[2].set_title("Survival = 1 − CIF", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.caption(f"📦 {artifacts.provenance(res)} — predictions from the tuned "
           "DeepHitSingle, on real test-set innings it never saw in training.")

st.markdown(
    f"""
**Read the three panels left to right — they are the same information, three
ways.** The CIF is the PMF's running total; survival is that flipped upside
down.

One detail worth noticing rather than glossing over: this batter's PMF bars sum
to **{pmf.sum():.3f}**, not exactly 1. The missing **{1 - pmf.sum():.3f}** is
the probability he is **still batting after the final bin** — the model's way of
saying "he might last longer than our time axis goes". That leftover mass is
exactly the height at which the survival curve (right panel) flattens out
instead of reaching zero.

Notice the shape of the PMF: for this batter the model puts its highest
probability mass in the **early bins** — a batter is most likely to be out soon
after arriving, because he has to survive the early bins to reach the later
ones. This is not a bug; it's the difference between a **PMF** (unconditional:
*"when will he go?"*) and a **hazard** (conditional: *"given he's still in, is
he about to go?"*). The hazard on the previous page was near-flat; the PMF
declines. Both are true, and they describe the same reality.
"""
)

st.header("2 · Where does the flexibility come from?")
st.markdown(
    r"""
Here is the payoff of the whole design, and the answer to *"why bother?"*

In DeepSurv, two batters' survival curves are locked together:
$S_1(t) = S_0(t)^{\exp(\hat h_1)}$ and $S_2(t) = S_0(t)^{\exp(\hat h_2)}$ — both
are **powers of the same baseline curve**. They can never cross. The model has
one curve and a per-player exponent.

In DeepHit, each batter gets **his own $K$ softmax outputs**. Nothing ties them
to a shared baseline. If the network wants to say *"this opener is at high risk
for 10 balls and then becomes the safest man on the field"*, it simply puts mass
in the early bins and almost none later. **Two batters' curves can cross freely**
— which, as the Schoenfeld test on the first page proved, is exactly what this
data demands.

The cost is equally real, and you should be able to state it:

- **More parameters.** The output layer is $K$ wide, not 1. With small survival
  datasets that is a genuine overfitting risk (hence dropout, and the small
  networks we use here).
- **You lose the hazard ratio.** A Cox coefficient gives you
  $e^{\beta} = 1.35$: *"each extra unit multiplies the hazard by 1.35"* —
  interpretable, reportable, testable. DeepHit gives you a probability
  distribution and no such summary. For a clinician or a selector, that is a
  real loss.
- **Resolution is capped by your bins.** You can never distinguish two events
  inside the same bin. Discretisation is *lossy*, permanently.
"""
)

st.header("3 · In code — softmax to survival, by hand")
show_example(
    '''import numpy as np

# Pretend the network just produced these 8 raw outputs (logits) for one batter
logits = np.array([2.4, 1.9, 1.1, 0.6, 0.1, -0.4, -0.9, -1.3])

# 1. SOFTMAX -> a probability mass function over the 8 time bins
exp = np.exp(logits - logits.max())      # subtract the max: overflow-safe
pmf = exp / exp.sum()
print("PMF:        ", pmf.round(3))
print("sums to:    ", pmf.sum().round(6), "  <- it IS a distribution")

# 2. CIF: the running total
cif = np.cumsum(pmf)
print("CIF:        ", cif.round(3))

# 3. SURVIVAL: whatever is left
surv = 1 - cif
print("Survival:   ", surv.round(3))

# The hazard is a DERIVED quantity now, not the thing we modelled:
#   h_k = P(out in bin k | survived to bin k) = pmf_k / S(k-1)
prev_surv = np.r_[1.0, surv[:-1]]
hazard = pmf / prev_surv
print("hazard:     ", hazard.round(3))
print()
print("note the hazard RISES even though the PMF FALLS - fewer batters are")
print("left to be dismissed, so each remaining one is at higher per-bin risk.")''',
    """
- `np.exp(logits - logits.max())` — the numerically-safe softmax. Subtracting the max prevents `exp` overflowing (the same trick as `logsumexp` in Section 6). Never write a raw `np.exp(logits)` on unbounded values.
- `pmf.sum()` printing `1.0` — the guarantee softmax gives you for free, and the reason no separate normalisation step is needed.
- `np.cumsum(pmf)` — the CIF. One line.
- `1 - cif` — survival. One more line. **There is no baseline hazard anywhere in this code**; compare with DeepSurv, where you must call `compute_baseline_hazards()` on the training set before you can produce a single survival curve.
- The final `hazard = pmf / prev_surv` block — this is worth staring at. The *hazard* is now something you **derive** from the PMF, not something you assume the shape of. DeepHit inverted the modelling relationship: Cox models the hazard and derives the distribution; DeepHit models the distribution and derives the hazard.
""",
)

guided_sandbox(
    key="h4",
    steps="""
1. **Step 1** — write `softmax(logits)` yourself (remember to subtract the max
   before exponentiating). Apply it to the `logits` in the editor and confirm
   the result sums to 1.
2. **Step 2** — from that PMF, compute the CIF (`np.cumsum`) and the survival
   curve (`1 - cif`). Print all three.
3. **Step 3** — derive the discrete hazard: `pmf[k] / S(k-1)`, where `S(-1)=1`.
   Print it. Does the hazard rise while the PMF falls? Explain to yourself why
   that is not a contradiction.
4. **Step 4 (stretch)** — invent a batter the model *fears*: make the logits
   large in the **early** bins and tiny later. Then invent a well-set batter:
   the reverse. Print both survival curves and confirm they **cross** — the
   thing a Cox model can never do.
""",
    setup_code='''import numpy as np

logits = np.array([2.4, 1.9, 1.1, 0.6, 0.1, -0.4, -0.9, -1.3])

# Step 1: define softmax(logits); confirm it sums to 1


# Step 2: PMF -> CIF -> survival


# Step 3: the derived discrete hazard


# Step 4 (stretch): two batters whose survival curves CROSS
''',
    solution_code='''import numpy as np

logits = np.array([2.4, 1.9, 1.1, 0.6, 0.1, -0.4, -0.9, -1.3])

def softmax(z):
    e = np.exp(z - z.max())      # overflow-safe
    return e / e.sum()

pmf = softmax(logits)
print("PMF:     ", pmf.round(3), " sums to", pmf.sum().round(6))

cif = np.cumsum(pmf)
surv = 1 - cif
print("CIF:     ", cif.round(3))
print("survival:", surv.round(3))

prev = np.r_[1.0, surv[:-1]]
hazard = pmf / prev
print("hazard:  ", hazard.round(3))
print("PMF falls but hazard rises: fewer batters remain at risk, so each")
print("survivor's per-bin chance of going is higher. Both are correct.")

nervous = softmax(np.array([3.0, 2.5, 1.0, 0.0, -1.0, -2.0, -2.5, -3.0]))
settled = softmax(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 1.2, 1.4, 1.5]))
s_nervous = 1 - np.cumsum(nervous)
s_settled = 1 - np.cumsum(settled)
print("\\n bin | nervous S(t) | settled S(t)")
for k in range(8):
    star = "  <- they CROSS" if (s_nervous[k] - s_settled[k]) * (
        s_nervous[0] - s_settled[0]) < 0 else ""
    print(f" {k:3d} | {s_nervous[k]:12.3f} | {s_settled[k]:11.3f}{star}")
print("\\nA Cox/DeepSurv model CANNOT produce crossing curves. DeepHit can.")''',
)
