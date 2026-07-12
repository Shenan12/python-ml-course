import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree

from utils.mockdata import patients_split
from utils.sandbox import guided_sandbox, show_example

st.title("🌳 Decision Trees")
st.markdown(
    """
A decision tree plays 20 Questions with your data: *"Is biomarker A above
5.1? If yes, is B above 3.8? …"* — a flowchart of yes/no questions ending in
a diagnosis. Two things make it beloved: you can **read** the fitted model
(clinicians can audit it), and it's the building block of Random Forests and
gradient boosting, the workhorses of applied ML (next two pages).

The algorithm needs to answer just one question, repeatedly: **which single
yes/no split is best?** "Best" = the split whose two resulting groups are as
*pure* as possible (each dominated by one class). So first we need a number
that measures impurity.
"""
)

st.header("1 · Measuring messiness: Gini impurity and entropy")
st.markdown(
    r"""
Take a group of patients where a fraction $p$ have the disease. Two standard
impurity scores:

- **Gini impurity**: $\;2p(1-p)$ — the chance that two patients drawn at
  random from the group have different labels.
- **Entropy**: $\;-p\log_2 p - (1-p)\log_2(1-p)$ — the average number of
  yes/no questions of information you're missing (in bits).

Both are 0 for a pure group and peak at a 50/50 coin-flip group. Drag $p$:
"""
)
p = st.slider("fraction of the group with the disease, p", 0.0, 1.0, 0.3,
              0.01)


def gini(p):
    return 2 * p * (1 - p)


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


ps = np.linspace(0, 1, 201)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(ps, gini(ps), color="#1565c0", linewidth=2, label="Gini  2p(1−p)")
ax.plot(ps, entropy(ps), color="#e65100", linewidth=2, label="entropy (bits)")
ax.scatter([p, p], [gini(p), entropy(p)], s=90, color=["#1565c0", "#e65100"],
           zorder=5)
ax.annotate(f"Gini = {gini(p):.3f}", (p, gini(p)),
            textcoords="offset points", xytext=(10, -14), fontsize=9)
ax.annotate(f"entropy = {entropy(p):.3f}", (p, entropy(p)),
            textcoords="offset points", xytext=(10, 8), fontsize=9)
ax.set_xlabel("p (fraction with disease)")
ax.set_ylabel("impurity")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    "The two curves have the same shape and almost always pick the same "
    "splits; Gini is sklearn's default (no logarithm to compute). We'll use "
    "Gini from here, and you'll compute entropy yourself in the sandbox."
)

X_train, X_val, y_train, y_val = patients_split()

st.header("2 · How a split is chosen: scan every threshold, live")
st.markdown(
    """
To choose its question, the tree tries **every feature and every possible
threshold**, scores each candidate split by the *weighted average impurity*
of the two groups it creates (weighted by group size), and keeps the
minimum. Here is that scan, for real, on our 80 training patients — pick a
feature and drag the candidate threshold:
"""
)
feat_name = st.radio("feature to split on", ["biomarker A", "biomarker B"],
                     horizontal=True)
feat = 0 if feat_name.endswith("A") else 1
thresh = st.slider(f"candidate question:  “is {feat_name} ≤ t ?”", 0.5, 9.5,
                   5.0, 0.05)


def split_score(f, t):
    left, right = y_train[X_train[:, f] <= t], y_train[X_train[:, f] > t]
    if len(left) == 0 or len(right) == 0:
        return gini(y_train.mean()), None, None
    score = (len(left) * gini(left.mean())
             + len(right) * gini(right.mean())) / len(y_train)
    return score, left, right


ts = np.linspace(0.5, 9.5, 361)
curve = [split_score(feat, t)[0] for t in ts]
curve_other = [split_score(1 - feat, t)[0] for t in ts]
best_t = ts[int(np.argmin(curve))]
best_other = ts[int(np.argmin(curve_other))]
your_score, left, right = split_score(feat, thresh)

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
        m = y_train == cls
        ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=28,
                   edgecolor="white")
    if feat == 0:
        ax.axvline(thresh, color="#2e7d32", linewidth=2.5)
        ax.axvline(best_t, color="#2e7d32", linewidth=1.2, linestyle=":")
    else:
        ax.axhline(thresh, color="#2e7d32", linewidth=2.5)
        ax.axhline(best_t, color="#2e7d32", linewidth=1.2, linestyle=":")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("biomarker A")
    ax.set_ylabel("biomarker B")
    ax.set_title("your split (solid) vs the best (dotted)", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(ts, curve, color="#2e7d32", linewidth=2,
            label=f"split on {feat_name}")
    ax.plot(ts, curve_other, color="#b0bec5", linewidth=1.5,
            label="the other feature")
    ax.scatter([thresh], [your_score], s=90, color="#e65100", zorder=5,
               label=f"your t = {thresh:.2f}")
    ax.scatter([best_t], [min(curve)], s=140, marker="*", color="#2e7d32",
               zorder=5, label=f"best t = {best_t:.2f}")
    ax.set_xlabel("threshold t")
    ax.set_ylabel("weighted Gini after the split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("the scan the algorithm performs", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

if left is not None:
    st.markdown(
        f"Your split: **left group** {len(left)} patients "
        f"({left.mean():.0%} diseased, Gini {gini(left.mean()):.3f}) · "
        f"**right group** {len(right)} patients ({right.mean():.0%} "
        f"diseased, Gini {gini(right.mean()):.3f}) → weighted score "
        f"**{your_score:.4f}**. The parent group's Gini was "
        f"{gini(y_train.mean()):.3f}; the drop is the *information gained* "
        "by asking your question. The tree asks the question with the "
        "biggest drop — then repeats the whole scan inside each group, "
        "recursively. That's the entire training algorithm."
    )

st.header("3 · The grown tree — as an actual tree")
depth = st.slider("max_depth (how many questions deep may it go?)", 1, 6, 2)
tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
tree.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10, 2.2 + 1.5 * depth))
plot_tree(tree, feature_names=["biomarker A", "biomarker B"],
          class_names=["healthy", "disease"], filled=True, rounded=True,
          fontsize=8, ax=ax, impurity=True)
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Each box: the question asked, the node's Gini, how many training "
    "patients reached it, their class counts, and the majority diagnosis. "
    "Blue = healthy-leaning, orange = disease-leaning; deeper colour = "
    "purer. Reading top-down IS running the model. Note the root's split "
    "matches the ⭐ the scan found above."
)

gx, gy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
zz = tree.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
            colors=["#bbdefb", "#ffcdd2"], alpha=0.75)
for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=26,
               edgecolor="white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_title(f"the same tree as decision regions (depth {depth}) — "
             "always rectangles!", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("training accuracy", f"{tree.score(X_train, y_train):.0%}")
c2.metric("validation accuracy", f"{tree.score(X_val, y_val):.0%}")
st.markdown(
    """
Every question is about *one* feature, so every boundary line is vertical or
horizontal — trees carve the world into **rectangles**. Crank the depth up
and watch the tree gerrymander slivers of rectangle around individual
training points: **depth is the tree's flexibility dial**, third verse of
the same overfitting song (degree, k, now depth).

Watch the two accuracy numbers as you do it. Training accuracy marches to
**100%** — by depth 5 the tree has memorised every training patient
perfectly. Validation accuracy does *not* follow it up; it peaks in the
middle of the range and then drifts back down, so all that extra depth
bought pure memorisation and nothing more. (On this easy two-blob data the
validation drop is mild — a few points. On messier, higher-dimensional data
the same mechanism costs you far more, which is exactly why the next two
pages exist: forests and boosting are both, at heart, defences against a
single tree's eagerness to memorise.)
"""
)

st.header("4 · In code")
show_example(
    '''import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X_train, y_train)

print(export_text(tree, feature_names=["biomarker A", "biomarker B"]))
print("validation accuracy:", tree.score(X_val, y_val))
print("feature importances:", tree.feature_importances_.round(3))''',
    """
- `DecisionTreeClassifier(max_depth=2, random_state=0)` — same sklearn pattern as KNN. `max_depth` is the flexibility hyperparameter; `random_state` fixes internal tie-breaks so results are reproducible.
- `export_text(...)` — prints the fitted flowchart as indented text: literally the rules, readable without any plotting.
- `tree.score(X_val, y_val)` — accuracy on held-out patients (`.score` on classifiers = accuracy).
- `tree.feature_importances_` — how much each feature's splits reduced impurity in total, normalised to sum to 1. A first taste of *model interpretation*; forests inherit this attribute too.
""",
)

guided_sandbox(
    key="c2",
    steps="""
1. **Step 1** — write `gini(p)` returning `2 * p * (1 - p)`, and
   `entropy(p)` using the formula above (`np.log2`; use
   `np.clip(p, 1e-12, 1 - 1e-12)` first so p=0 or 1 doesn't crash the log).
   Print both for p = 0.5 and p = 0.9.
2. **Step 2** — write `split_quality(feature, t)`: split `y_train` into the
   labels where `X_train[:, feature] <= t` and the rest, and return the
   size-weighted average of the two groups' Gini scores.
3. **Step 3** — print `split_quality(0, t)` for t in `[3, 4, 5, 6, 7]`.
   Which threshold on biomarker A is best? Compare with the root question
   of the fitted sklearn tree (printed by the setup code).
4. **Step 4 (stretch)** — fit trees with `max_depth` 1..8 and print each
   validation accuracy. Where does going deeper stop paying?
""",
    setup_code='''import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
sk_tree = DecisionTreeClassifier(max_depth=2, random_state=0)
sk_tree.fit(X_train, y_train)
print("sklearn's tree for reference:")
print(export_text(sk_tree, feature_names=["biomarker A", "biomarker B"]))

# Step 1: define gini(p) and entropy(p); print both at p=0.5 and p=0.9


# Step 2: define split_quality(feature, t)


# Step 3: try t in [3, 4, 5, 6, 7] on biomarker A (feature 0)


# Step 4 (stretch): validation accuracy for max_depth 1..8
''',
    solution_code='''import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()

def gini(p):
    return 2 * p * (1 - p)

def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

for p in [0.5, 0.9]:
    print(f"p={p}: gini={gini(p):.3f}  entropy={entropy(p):.3f}")

def split_quality(feature, t):
    mask = X_train[:, feature] <= t
    left, right = y_train[mask], y_train[~mask]
    if len(left) == 0 or len(right) == 0:
        return gini(y_train.mean())
    return (len(left) * gini(left.mean())
            + len(right) * gini(right.mean())) / len(y_train)

for t in [3, 4, 5, 6, 7]:
    print(f"split 'biomarker A <= {t}': weighted gini "
          f"{split_quality(0, t):.4f}")

for d in range(1, 9):
    tr = DecisionTreeClassifier(max_depth=d, random_state=0)
    tr.fit(X_train, y_train)
    print(f"depth {d}: train {tr.score(X_train, y_train):.0%}  "
          f"val {tr.score(X_val, y_val):.0%}")''',
)
