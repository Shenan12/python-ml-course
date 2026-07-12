import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.mockdata import patients_split
from utils.sandbox import guided_sandbox, show_example

st.title("🌲🌲 Bagging & Random Forests")
st.markdown(
    """
On the trees page, depth-6 trees memorised the training data — they have
**high variance**: regrow the tree on a slightly different sample and you get
a wildly different flowchart. Averaging is the statistician's oldest variance
cure (the variance of a mean of $B$ independent measurements falls like
$1/B$), and that's the whole trick here: **train many overfit trees, let
them vote, and the noise cancels while the signal doesn't.**

Two ingredients:

1. **Bagging** (*bootstrap aggregating*) — we only have one training set, so
   where do "many samples" come from? Draw B **bootstrap samples**: each is
   80 patients drawn *with replacement* from our 80. Each tree trains on its
   own resample, so each overfits *differently*.
2. **Random feature choice** — bagged trees still tend to agree too much
   (they all love the best split). A **Random Forest** additionally offers
   each split only a random subset of features, forcing trees to find
   different routes to the answer. Less correlated trees → averaging works
   better.
"""
)

X_train, X_val, y_train, y_val = patients_split()

st.header("1 · What a bootstrap sample actually looks like")
st.markdown(
    "Below: 20 of our training patients, and how many times each one lands "
    "in a bootstrap draw of size 20. Change the seed — every draw misses "
    "some patients entirely (on average ≈ 37% of them, the famous "
    "**out-of-bag** points) and picks others two or three times."
)
seed = st.slider("bootstrap draw seed", 0, 20, 0)
r = np.random.default_rng(seed)
draw = r.integers(0, 20, 20)
counts = np.bincount(draw, minlength=20)

fig, ax = plt.subplots(figsize=(9.5, 2.4))
for i in range(20):
    c = counts[i]
    colour = ["#eceff1", "#a5d6a7", "#66bb6a", "#2e7d32", "#1b5e20"][min(c, 4)]
    ax.add_patch(plt.Rectangle((i, 0), 0.9, 1.2, facecolor=colour,
                               edgecolor="#546e7a"))
    ax.text(i + 0.45, 0.6, f"×{c}", ha="center", va="center", fontsize=10,
            color="white" if c >= 2 else "#78909c",
            fontweight="bold" if c else "normal")
    ax.text(i + 0.45, -0.25, f"p{i}", ha="center", fontsize=7,
            color="#78909c")
n_oob = int((counts == 0).sum())
ax.set_title(f"one bootstrap draw: {20 - n_oob} distinct patients in, "
             f"{n_oob} left out-of-bag", fontsize=10)
ax.set_xlim(-0.3, 20.2)
ax.set_ylim(-0.55, 1.5)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Why ≈ 37% miss out: each of the 20 draws misses patient *i* with "
    "probability 19/20, so P(never drawn) = (19/20)²⁰ ≈ 0.358 — and for "
    "large n it tends to 1/e ≈ 0.368. The out-of-bag patients are free "
    "validation data for that tree — sklearn exploits this (`oob_score`, "
    "in the code below)."
)

st.header("2 · Many shaky trees → one steady forest")
st.markdown(
    "Every boundary here is fitted live. Grey lines: individual "
    "**fully-grown** (unlimited depth) trees, each trained on its own "
    "bootstrap sample — watch how much they disagree. Colour: their "
    "majority vote."
)
c1, c2 = st.columns(2)
n_trees = c1.select_slider("number of trees B", [1, 5, 25, 100], value=5)
show_individuals = c2.slider("…of which, show individual boundaries for",
                             0, min(10, n_trees), min(5, n_trees))

rng = np.random.default_rng(0)
gx, gy = np.meshgrid(np.linspace(0, 10, 150), np.linspace(0, 10, 150))
grid = np.column_stack([gx.ravel(), gy.ravel()])

tree_preds = []
for i in range(n_trees):
    idx = rng.integers(0, len(X_train), len(X_train))   # bootstrap
    t = DecisionTreeClassifier(random_state=i).fit(X_train[idx], y_train[idx])
    tree_preds.append(t.predict(grid))
tree_preds = np.array(tree_preds)
vote_share = tree_preds.mean(axis=0).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contourf(gx, gy, vote_share, levels=[-0.01, 0.5, 1.01],
            colors=["#bbdefb", "#ffcdd2"], alpha=0.75)
for i in range(show_individuals):
    ax.contour(gx, gy, tree_preds[i].reshape(gx.shape), levels=[0.5],
               colors="#78909c", linewidths=0.7, alpha=0.8)
ax.contour(gx, gy, vote_share, levels=[0.5], colors="#37474f", linewidths=2.5)
for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=26,
               edgecolor="white", zorder=3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_title(f"{n_trees} bagged tree(s): thin grey = individuals, "
             "thick = their vote", fontsize=10)
st.pyplot(fig)
plt.close(fig)

# honest accuracy comparison, computed live
single = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
vote_val = np.zeros(len(X_val))
rng2 = np.random.default_rng(0)
for i in range(n_trees):
    idx = rng2.integers(0, len(X_train), len(X_train))
    t = DecisionTreeClassifier(random_state=i).fit(X_train[idx], y_train[idx])
    vote_val += t.predict(X_val)
bag_acc = np.mean((vote_val / n_trees > 0.5).astype(int) == y_val)

c1, c2 = st.columns(2)
c1.metric("1 fully-grown tree (val accuracy)",
          f"{single.score(X_val, y_val):.0%}")
c2.metric(f"vote of {n_trees} bagged tree(s)", f"{bag_acc:.0%}")
st.markdown(
    """
Step B up from 1 → 5 → 25 → 100 and watch the **boundary**, which is where
the mechanism shows: the individual grey lines stay as jagged and neurotic
as ever — **each tree still overfits!** — but their vote smooths into a
stable, sensible frontier, and the little islands that one tree wrapped
around stray points get outvoted away. Averaging killed the variance without
touching the (low) bias of deep trees. That's the entire philosophy.

An honesty note about the accuracy numbers above: our two-blob data is easy
enough that a single tree already scores well, so the vote's accuracy gain
is small and can even wobble *down* a point or two between B values — with
only 40 validation patients, one patient flipping is 2.5%. Don't read a
trend into that noise; the reliable, visible win here is the **stability of
the boundary**, and on harder data (more features, more noise) the accuracy
gain becomes substantial too. What is genuinely guaranteed: unlike boosting
(next page), **more trees never makes a forest overfit** — B is a "more is
better until your patience runs out" knob, not a flexibility dial.
"""
)

st.header("3 · The real thing: `RandomForestClassifier`")
show_example(
    '''from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()

forest = RandomForestClassifier(
    n_estimators=200,       # B: how many trees
    max_features="sqrt",    # each split sees only sqrt(n_features) features
    oob_score=True,         # free validation from out-of-bag patients
    random_state=0,
)
forest.fit(X_train, y_train)

single = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("single deep tree, validation:", single.score(X_val, y_val))
print("random forest,    validation:", forest.score(X_val, y_val))
print("out-of-bag estimate:         ", round(forest.oob_score_, 3))
print("feature importances:", forest.feature_importances_.round(3))
print("P(disease) for patient [4.4, 6.0]:",
      forest.predict_proba([[4.4, 6.0]])[0][1])''',
    """
- `n_estimators=200` — the number of trees. 100–500 is a common default range; more is slower but never *hurts* accuracy.
- `max_features="sqrt"` — the "random" in Random Forest: at every split, each tree may only consider a random √(number of features) of the features. (With our 2 features this does little; with 30 features it's transformative.)
- `oob_score=True` — sklearn scores each patient using only the trees that *didn't* see them (the out-of-bag trick from the visualization) — an honest accuracy estimate without spending your validation set. Compare `oob_score_` with the validation number: they should roughly agree.
- `feature_importances_` — impurity-based importances averaged over all trees, more stable than a single tree's.
- `predict_proba` — the share of trees voting "disease": 200 opinions compressed into a probability.
- Practical notes: forests barely need feature scaling (trees only compare thresholds), tolerate messy data well, and their main costs are memory/speed and losing the single tree's read-it-like-a-flowchart interpretability.
""",
)

guided_sandbox(
    key="c5",
    steps="""
1. **Step 1** — fit one `DecisionTreeClassifier(random_state=0)` (no depth
   limit) on the full training data and print its train and validation
   accuracy. Diagnose it: which of the two numbers screams overfitting?
2. **Step 2** — build your own bagger: 25 times, draw a bootstrap sample
   (`idx = rng.integers(0, 80, 80)`), fit a tree on `X_train[idx],
   y_train[idx]`, and collect each tree's `predict(X_val)` into a list.
3. **Step 3** — stack the collected predictions (`np.array`), average over
   trees (`axis=0`), threshold at 0.5, and print the vote's validation
   accuracy next to the single tree's.
4. **Step 4 (stretch)** — compare with
   `RandomForestClassifier(n_estimators=25, random_state=0)`, then try
   `n_estimators=200`. And check `oob_score=True` roughly agrees with the
   validation accuracy.
""",
    setup_code='''import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
rng = np.random.default_rng(0)
print("data ready:", X_train.shape, "train,", X_val.shape, "val")

# Step 1: one fully-grown tree - train vs val accuracy


# Step 2: 25 bootstrap trees, collect predictions on X_val


# Step 3: average the votes, threshold at 0.5, print val accuracy


# Step 4 (stretch): RandomForestClassifier comparison + oob_score
''',
    solution_code='''import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
rng = np.random.default_rng(0)

single = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print(f"single tree: train {single.score(X_train, y_train):.0%}, "
      f"val {single.score(X_val, y_val):.0%}  <- perfect train = memorised")

preds = []
for i in range(25):
    idx = rng.integers(0, len(X_train), len(X_train))
    t = DecisionTreeClassifier(random_state=i)
    t.fit(X_train[idx], y_train[idx])
    preds.append(t.predict(X_val))
vote = (np.array(preds).mean(axis=0) > 0.5).astype(int)
print(f"my 25-tree bagger: val {np.mean(vote == y_val):.0%}")

for B in [25, 200]:
    rf = RandomForestClassifier(n_estimators=B, oob_score=True,
                                random_state=0).fit(X_train, y_train)
    print(f"RandomForest B={B:3d}: val {rf.score(X_val, y_val):.0%}, "
          f"oob estimate {rf.oob_score_:.0%}")''',
)
