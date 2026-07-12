import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from utils.sandbox import guided_sandbox, show_example

st.title("✂️ Train / Validation / Test Splits")
st.markdown(
    """
Here is the most important honesty rule in machine learning: **a model must
be judged on data it has never seen.** A student who memorises last year's
exam answers hasn't learned statistics — and a model that memorises its
training data hasn't learned the pattern. Checking a model on the same data
it learned from is exactly that student marking their own memorised paper.

So we split the data into three piles, with strict jobs:

| Pile | Exam analogy | Used for | Used how often |
|---|---|---|---|
| **Training set** | homework & practice | fitting the parameters | constantly |
| **Validation set** | mock exam | *choosing between* models & settings | many times |
| **Test set** | the real final exam | one final honest score | **once, at the very end** |

Why three and not two? Because if you tweak your model 50 times and keep the
version with the best validation score, you have quietly *fitted your
choices* to the validation set — its score is now flattering. The untouched
test set is the only pile with no fingerprints on it. (In Sections 6–7
you'll see DeepSurv and DeepHit judged exactly this way.)
"""
)

st.header("1 · Watch a split happen — and watch it go wrong")
st.markdown(
    """
40 mock students, and here's the trap: **they arrive sorted by mark**
(weakest first), which is how real data often turns up — sorted by date, by
hospital, by severity. Each square is one student, in the order they sit in
the dataset. Choose the split sizes, then try turning shuffling **off**.
"""
)

marks_sorted = np.sort(np.clip(np.random.default_rng(8).normal(64, 13, 40),
                               20, 98)).round(0)

c1, c2, c3 = st.columns(3)
train_pct = c1.slider("training %", 40, 80, 60, 5)
val_pct = c2.slider("validation %", 10, 30, 20, 5)
seed = c3.number_input("shuffle seed", 0, 999, 0)
shuffle = st.toggle("shuffle before splitting", value=True)
test_pct = 100 - train_pct - val_pct

n = len(marks_sorted)
order = (np.random.default_rng(int(seed)).permutation(n) if shuffle
         else np.arange(n))
n_train = int(n * train_pct / 100)
n_val = int(n * val_pct / 100)
assignment = np.empty(n, dtype=object)
assignment[order[:n_train]] = "train"
assignment[order[n_train:n_train + n_val]] = "val"
assignment[order[n_train + n_val:]] = "test"

COLOURS = {"train": "#42a5f5", "val": "#ffb300", "test": "#66bb6a"}
fig, ax = plt.subplots(figsize=(10, 1.9))
for i in range(n):
    ax.add_patch(Rectangle((i, 0), 0.92, 1.4,
                           facecolor=COLOURS[assignment[i]],
                           edgecolor="white"))
    ax.text(i + 0.45, 0.7, f"{marks_sorted[i]:.0f}", ha="center",
            va="center", fontsize=6.5, rotation=90, color="#263238")
ax.text(0, 1.75, "dataset in stored order → sorted weakest to strongest",
        fontsize=9, color="#546e7a")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in COLOURS.values()]
ax.legend(handles, [f"train ({n_train})", f"val ({n_val})",
                    f"test ({n - n_train - n_val})"],
          loc="lower right", fontsize=8, ncol=3, bbox_to_anchor=(1.0, -0.5))
ax.set_xlim(-0.3, n + 0.3)
ax.set_ylim(-0.2, 2.1)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

stats = {s: marks_sorted[assignment == s].mean()
         for s in ["train", "val", "test"]}
c1, c2, c3 = st.columns(3)
c1.metric("train mean mark", f"{stats['train']:.1f}")
c2.metric("val mean mark", f"{stats['val']:.1f}")
c3.metric("test mean mark", f"{stats['test']:.1f}")

gap = max(stats.values()) - min(stats.values())
if not shuffle:
    st.error(
        f"**This is the disaster.** Unshuffled, the model trains only on "
        f"the weakest students (mean {stats['train']:.1f}) and is tested "
        f"only on the strongest (mean {stats['test']:.1f}) — a gap of "
        f"{gap:.1f} marks. The three piles are three *different* "
        "populations, so any score you compute is meaningless. Always "
        "shuffle (unless order itself matters, e.g. forecasting time series "
        "— then you split by time on purpose)."
    )
else:
    st.success(
        f"Shuffled, all three piles have similar means (spread of just "
        f"{gap:.1f} marks) — three fair samples of the same population. "
        "Try a few different seeds: the exact numbers wobble, but the piles "
        "stay comparable."
    )

st.header("2 · The standard tool: `train_test_split`")
show_example(
    '''import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(8)
X = rng.uniform(0, 12, (40, 2))          # 40 students, 2 features each
y = np.clip(38 + 3.6 * X[:, 0] + rng.normal(0, 6, 40), 0, 100)

# Split once: 60% train, 40% leftover
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Split the leftover in half: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, random_state=42)

print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)
print("train mean mark:", y_train.mean().round(1))
print("val   mean mark:", y_val.mean().round(1))
print("test  mean mark:", y_test.mean().round(1))''',
    """
- `from sklearn.model_selection import train_test_split` — your first import from **scikit-learn** (`sklearn`), the standard classical-ML library that Section 3 lives in.
- `X = rng.uniform(0, 12, (40, 2))` — note the shape convention you'll now see everywhere: `X` is a 2-D array of `(number_of_samples, number_of_features)`, capital X; `y` is the 1-D answers, lowercase.
- `train_test_split(X, y, test_size=0.4, random_state=42)` — shuffles (that's the default) and splits both `X` and `y` *consistently*, so each student's features stay glued to their mark. It returns four pieces — unpacked in one line, tuple-style.
- `random_state=42` — the seed, so your split (and your dissertation's!) is reproducible.
- The function only makes *two* piles, so we call it **twice** to get three — the second call halves the 40% leftover into val and test.
- The three printed means being close is the same fairness check as the coloured strip above.
""",
)

st.header("3 · The rules of the three piles")
st.markdown(
    """
1. **Fit on train, only.** The model's parameters never see val or test.
2. **Compare on validation.** Trying degree-3 vs degree-9? Learning rate
   0.01 vs 0.1? The validation score is the referee. (Next two pages use it
   for exactly this.)
3. **Report on test, once.** After all decisions are final. If you then go
   back and tweak, the test set is burnt — its innocence is spent.
4. **Beware leakage.** *Any* information flowing from val/test into training
   is cheating, even sneakily: e.g. standardizing using the mean of the
   *whole* dataset lets the test set's statistics leak into training.
   Compute preprocessing statistics on the training pile only, then apply
   them to the others.

A note for your dissertation: with small medical datasets a single split can
be unlucky. The fix — **cross-validation**, rotating which pile is held out —
is covered with the model-evaluation page in Section 3.
"""
)

guided_sandbox(
    key="m4",
    steps="""
1. **Step 1** — make a shuffled index order with
   `np.random.default_rng(0).permutation(len(marks))`.
2. **Step 2** — slice that order into three index groups: first 24 for
   train, next 8 for validation, last 8 for test.
3. **Step 3** — use the index groups to pull marks out (e.g.
   `marks[train_idx]`) and print each pile's `len` and `.mean().round(1)` —
   confirm the means are similar.
4. **Step 4 (stretch)** — repeat WITHOUT shuffling (`np.arange(len(marks))`
   as the order). How far apart are the means now? You've reproduced the
   red-box disaster above.
""",
    setup_code='''import numpy as np

# 40 mock student marks, arriving SORTED (weakest first):
marks = np.sort(np.clip(np.random.default_rng(8).normal(64, 13, 40),
                        20, 98)).round(0)
print("first five:", marks[:5], "  last five:", marks[-5:])

# Step 1: build a shuffled order of the indices 0..39


# Step 2: slice it -> train_idx (24), val_idx (8), test_idx (8)


# Step 3: print size and mean mark of each pile


# Step 4 (stretch): redo steps 2-3 with an UNSHUFFLED order
''',
    solution_code='''import numpy as np

marks = np.sort(np.clip(np.random.default_rng(8).normal(64, 13, 40),
                        20, 98)).round(0)

order = np.random.default_rng(0).permutation(len(marks))
train_idx, val_idx, test_idx = order[:24], order[24:32], order[32:]

for name, idx in [("train", train_idx), ("val", val_idx),
                  ("test", test_idx)]:
    print(f"{name:5}: n={len(idx):2}  mean={marks[idx].mean().round(1)}")

print("\\nunshuffled (the disaster):")
order = np.arange(len(marks))
train_idx, val_idx, test_idx = order[:24], order[24:32], order[32:]
for name, idx in [("train", train_idx), ("val", val_idx),
                  ("test", test_idx)]:
    print(f"{name:5}: n={len(idx):2}  mean={marks[idx].mean().round(1)}")''',
)
