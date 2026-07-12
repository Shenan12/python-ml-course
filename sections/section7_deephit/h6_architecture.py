import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils import cricket
from utils.sandbox import guided_sandbox, show_example

st.title("🏗️ Architecture — and a Word on Competing Risks")

st.header("1 · The network, and how it differs from DeepSurv")
st.markdown(
    r"""
Everything you've built over the last three pages assembles into one picture.
DeepHit is a feed-forward network like any other — the interesting parts are at
the **two ends**.
"""
)

show_multi = st.toggle("show the full competing-risks architecture "
                       "(the paper's general form)", value=False)

fig, ax = plt.subplots(figsize=(10.5, 4.2))


def box(x, y, w, h, text, face, sub=None, fs=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="#37474f",
                           linewidth=1.6))
    ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), text, ha="center",
            va="center", fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
                fontsize=7.5, color="#546e7a")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color="#78909c",
                                 linewidth=1.6))


box(0.2, 1.6, 1.7, 1.0, "features x", "#e3f2fd",
    "20 cricket features", fs=9)
arrow(1.95, 2.1, 2.55, 2.1)
box(2.6, 1.5, 1.9, 1.2, "shared layers", "#fff3e0",
    "the 'trunk'\n32 → 32, ReLU", fs=9)

if not show_multi:
    arrow(4.55, 2.1, 5.15, 2.1)
    box(5.2, 1.5, 1.9, 1.2, "output layer", "#ede7f6", "K = 20 neurons", fs=9)
    arrow(7.15, 2.1, 7.75, 2.1)
    box(7.8, 1.5, 2.3, 1.2, "SOFTMAX", "#ffcdd2",
        "PMF over 20 time bins\nΣ p_k = 1", fs=10)
    ax.text(5.3, 0.95, "one output per TIME BIN — not one risk score",
            fontsize=8, color="#c62828")
else:
    for i, (yy, nm) in enumerate([(2.75, "cause 1: bowled"),
                                  (1.55, "cause 2: caught"),
                                  (0.35, "cause 3: run out")]):
        arrow(4.55, 2.1, 5.15, yy + 0.4)
        box(5.2, yy, 1.7, 0.8, f"sub-network {i + 1}", "#ede7f6", nm, fs=8)
        arrow(6.95, yy + 0.4, 7.55, yy + 0.4)
    box(7.6, 0.35, 2.5, 3.2, "SOFTMAX\n(joint)", "#ffcdd2",
        "one PMF over\n(cause, time bin) pairs\nΣ over ALL = 1", fs=10)
    ax.text(5.2, 3.75, "one sub-network per competing cause",
            fontsize=8, color="#6a1b9a")

ax.set_xlim(0, 10.6)
ax.set_ylim(0, 4.1)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
| | DeepSurv | DeepHitSingle |
|---|---|---|
| **output width** | 1 | **K** (one per time bin) |
| **output activation** | none — a raw log-risk | **softmax** |
| **loss** | Cox negative log **partial** likelihood | **α·(likelihood) + (1−α)·(ranking)** |
| **needs a baseline hazard?** | **yes** — `compute_baseline_hazards()` on the training set before any curve | **no** — the curve *is* the output |
| **proportional hazards?** | **assumed** | **not assumed** |
| **can two survival curves cross?** | ❌ never | ✅ freely |
| **gives you a hazard ratio?** | ✅ (well, the network's risk score) | ❌ — you get a distribution instead |
| **time** | continuous | **discretised** (lossy) |
| **extra hyperparameters** | — | number of bins, bin scheme, α, σ |

**The trade in one sentence:** DeepHit buys freedom from the proportional-hazards
straitjacket, and pays for it with discretisation, four extra hyperparameters,
and the loss of an interpretable hazard ratio.
"""
)

st.header("2 · Competing risks — the thing DeepHit was really built for")
st.markdown(
    """
We've used **DeepHitSingle**: one event type (dismissal). But the paper's full
model handles **competing risks**, and cricket gives an unusually clean example
of what that means.

A batter isn't just "out" — he's out **in a particular way**, and those ways
compete with one another:
"""
)

try:
    df = cricket.load_raw()
except FileNotFoundError:
    st.error(
        "**The cricket dataset is missing** (`data/odi_batting_innings.csv`). "
        "If deploying, commit the `data/` folder; locally, run "
        "`python scripts/prepare_cricket_data.py`."
    )
    st.stop()
wt = (df[df.event == 1]["wicket_type"].value_counts(normalize=True) * 100)
wt = wt[wt >= 1.0]

fig, ax = plt.subplots(figsize=(8.5, 2.8))
ax.barh(wt.index[::-1], wt.values[::-1], color="#7e57c2", edgecolor="white")
for i, v in enumerate(wt.values[::-1]):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% of dismissals")
ax.set_title("how ODI batters actually get out (real Cricsheet data)",
             fontsize=10)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    r"""
**Why "competing"?** Because they are *mutually exclusive and they censor each
other*. If a batter is **caught** on ball 20, we will never observe when he
*would have been* bowled. The caught event has removed him from being at risk of
every other kind of dismissal. You cannot simply model "bowled" on its own and
treat catches as ordinary censoring — that would assume the two are independent,
which is plainly false (an aggressive batter attracts *both* more catches and
more stumpings, and the same shot selection drives both).

**What DeepHit does:** instead of one softmax over $K$ bins, it uses a **single
softmax over (cause, bin) pairs** — so all the probabilities, across every cause
and every time, sum to 1. Architecturally that means:

- one **shared trunk** (the features matter to every cause), then
- one **sub-network per cause** (each cause has its own patterns), then
- a **joint softmax** over the whole grid.

The output is the **cause-specific CIF**: $F_c(t \mid x)$ = *"the probability
this batter is out **by** ball $t$, **bowled**"*. The ranking term is then
applied **within each cause** separately.

The competing-risks alternative you'd meet in a classical statistics course is
the **Fine–Gray subdistribution hazard** model — worth knowing the name, since
it is DeepHit's linear ancestor exactly as Cox is DeepSurv's.

**For your dissertation:** it is enough to understand that DeepHitSingle is the
special case of DeepHit with one cause, and that the general model's contribution
is handling causes *jointly* rather than pretending they're independent. If your
data has one event type, `DeepHitSingle` is the right tool and the competing-risks
machinery is simply switched off.
"""
)
st.info(
    "**A cricket competing-risks project, if you ever want one:** predict the "
    "cause-specific CIF for bowled / caught / LBW / run out. It would let you "
    "ask genuinely interesting questions — *does a batter's dismissal-type risk "
    "profile shift as he settles?* (Almost certainly: early on you're bowled or "
    "LBW by movement; once set, you're more likely caught going for a big shot.) "
    "That is a competing-risks question, and it is exactly what DeepHit's full "
    "form is for."
)

st.header("3 · Building the model in code")
show_example(
    '''import torchtuples as tt
from pycox.models import DeepHitSingle
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
_, dtr, etr, _, _ = cricket.xy(tr, cols)

num_bins = 20

# 1. DISCRETISE: the label transform decides the output width.
#    It must be FITTED on the training durations before it knows anything.
labtrans = DeepHitSingle.label_transform(num_bins, scheme="quantiles")
labtrans.fit(dtr, etr)
print("output neurons needed:", labtrans.out_features)

# 2. THE NETWORK: an ordinary MLP - the only special thing is its WIDTH
net = tt.practical.MLPVanilla(
    in_features=len(cols),             # our cricket features
    num_nodes=[32, 32],                # a deliberately small trunk
    out_features=labtrans.out_features,  # K = one neuron per time bin
    batch_norm=True,
    dropout=0.1,
)
print("parameters:", sum(p.numel() for p in net.parameters()))

# 3. THE MODEL: softmax + the two-part loss live in here
model = DeepHitSingle(
    net,
    tt.optim.Adam(0.001),
    alpha=0.5,     # weight on the LIKELIHOOD term
    sigma=0.5,     # smoothness of the RANKING term
    duration_index=labtrans.cuts,   # so predictions come back in BALLS, not bin indices
)
print()
print("note: no output activation in the net - DeepHitSingle applies the")
print("softmax internally, exactly as BCEWithLogitsLoss did in Section 5.")''',
    """
- `labtrans.out_features` — **the discretisation decides the architecture.** Choose 20 bins and your output layer is 20 wide. This is why the binning page came first: it isn't preprocessing, it's a structural choice.
- `MLPVanilla(..., out_features=labtrans.out_features)` — otherwise an utterly ordinary network. Note `[32, 32]`: two small layers. On 15k rows with 20 features there is no reason to go bigger, and every reason not to (overfitting, and your laptop's fans).
- **No softmax in the network definition.** `DeepHitSingle` applies it inside the loss, for the same numerical-stability reason `BCEWithLogitsLoss` fused the sigmoid in Section 5. If you ever hand-roll this, don't apply softmax twice.
- `duration_index=labtrans.cuts` — bookkeeping that pays off at prediction time: it maps bin indices back to **balls faced**, so `predict_surv_df` returns a curve indexed by real cricket time rather than by bin number.
- `alpha` and `sigma` — the two knobs from the loss page, exposed right here on the constructor.
""",
)

guided_sandbox(
    key="h6",
    heavy=True,
    est="~10 s",
    steps="""
1. **Step 1** — build the `label_transform` with 20 quantile bins, fit it on
   the training durations, and print `out_features`. Then build the
   `MLPVanilla` with that width and print its parameter count.
2. **Step 2** — build the same network with 5 bins and with 50 bins. Print the
   parameter counts side by side. How much of the model's size is decided by
   your binning choice alone?
3. **Step 3** — construct the `DeepHitSingle` model and print `model.alpha` and
   `model.sigma`. Confirm they're the values you passed.
4. **Step 4 (stretch)** — using the real data, count how many dismissals of
   each `wicket_type` there are (`df[df.event==1].wicket_type.value_counts()`).
   Which causes are common enough that a competing-risks model could actually
   learn them, and which are too rare to bother with?
""",
    setup_code='''import numpy as np
import torchtuples as tt
from pycox.models import DeepHitSingle
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
_, dtr, etr, _, _ = cricket.xy(tr, cols)
print(f"{len(cols)} features, {len(tr):,} training innings")

# Step 1: label_transform(20, "quantiles") -> out_features -> build the net


# Step 2: compare parameter counts for 5, 20, 50 bins


# Step 3: build DeepHitSingle; print alpha and sigma


# Step 4 (stretch): dismissal types - which are common enough to model?
''',
    solution_code='''import numpy as np
import torchtuples as tt
from pycox.models import DeepHitSingle
from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
_, dtr, etr, _, _ = cricket.xy(tr, cols)

lab = DeepHitSingle.label_transform(20, scheme="quantiles")
lab.fit(dtr, etr)
print("out_features (= number of time bins):", lab.out_features)

net = tt.practical.MLPVanilla(len(cols), [32, 32], lab.out_features,
                              batch_norm=True, dropout=0.1)
print("parameters:", sum(p.numel() for p in net.parameters()))

print("\\nhow much does the BINNING decide the model size?")
for nb in [5, 20, 50]:
    lt = DeepHitSingle.label_transform(nb, scheme="quantiles")
    lt.fit(dtr, etr)
    n = tt.practical.MLPVanilla(len(cols), [32, 32], lt.out_features,
                                batch_norm=True, dropout=0.1)
    print(f"  {nb:2d} bins -> {lt.out_features:2d} outputs -> "
          f"{sum(p.numel() for p in n.parameters()):,} parameters")

model = DeepHitSingle(net, tt.optim.Adam(0.001), alpha=0.5, sigma=0.5,
                      duration_index=lab.cuts)
print(f"\\nalpha={model.alpha} (likelihood weight), "
      f"sigma={model.sigma} (ranking smoothness)")

raw = cricket.load_raw()
counts = raw[raw.event == 1].wicket_type.value_counts()
print("\\ndismissal types (competing risks):")
for cause, n in counts.items():
    verdict = "enough data" if n > 500 else "too rare to model well"
    print(f"  {cause:18} {n:6,}  ({n / counts.sum():5.1%})  {verdict}")''',
)
