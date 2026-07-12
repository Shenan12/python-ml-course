import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

from utils.mockdata import patients_split
from utils.sandbox import guided_sandbox, show_example

st.title("🏘️ K-Nearest Neighbors (KNN)")
st.markdown(
    """
Welcome to Section 3 — a tour of the classical algorithms that still do most
of the world's everyday ML work. We start with the most honest algorithm of
all. KNN's entire theory: **to classify a new case, find the k most similar
cases you've already seen, and let them vote.**

The running dataset for this whole section: 120 mock patients, two blood
biomarkers (A and B), labelled **healthy** or **disease** (seed 7, split 80
train / 40 validation). A new patient walks in with fresh biomarker values —
diagnose them by their neighbors:

- **Similar** = close in feature space. Distance is plain Euclidean —
  $\\sqrt{(A_1-A_2)^2 + (B_1-B_2)^2}$ — the straight-line distance on the
  scatter plot.
- **k** = how many neighbors get a vote. It's a hyperparameter (you now know
  exactly what that word means, and how to pick one).
- There is no "training" at all: the model *is* the stored dataset.
"""
)

X_train, X_val, y_train, y_val = patients_split()

st.header("1 · The vote, live")
st.markdown(
    "Drag the new patient around and change k. The circle encloses exactly "
    "the k nearest training patients (real distances, computed live); the "
    "vote is shown below."
)
c1, c2, c3 = st.columns(3)
qa = c1.slider("new patient: biomarker A", 0.0, 10.0, 5.2, 0.1)
qb = c2.slider("new patient: biomarker B", 0.0, 10.0, 4.6, 0.1)
k = c3.select_slider("k (number of voting neighbors)",
                     [1, 3, 5, 9, 15, 25, 79], value=5)

query = np.array([qa, qb])
dists = np.sqrt(((X_train - query) ** 2).sum(axis=1))   # real distances
nearest = np.argsort(dists)[:k]
votes_disease = int(y_train[nearest].sum())
votes_healthy = k - votes_disease
prediction = "disease" if votes_disease > votes_healthy else "healthy"
radius = dists[nearest].max()

fig, ax = plt.subplots(figsize=(7.5, 6))
for cls, colour, label in [(0, "#42a5f5", "healthy (train)"),
                           (1, "#ef5350", "disease (train)")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=36, label=label,
               edgecolor="white", zorder=3)
for i in nearest:
    ax.plot([qa, X_train[i, 0]], [qb, X_train[i, 1]], color="#9e9e9e",
            linewidth=0.9, zorder=2)
    ax.scatter(*X_train[i], s=130, facecolor="none", edgecolor="#2e7d32",
               linewidth=1.8, zorder=4)
ax.add_patch(plt.Circle((qa, qb), radius, fill=False, color="#2e7d32",
                        linestyle="--", linewidth=1.4))
ax.scatter([qa], [qb], marker="*", s=380, c="#ffd600", edgecolor="black",
           zorder=6, label="the NEW patient")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"the {k} nearest neighbors and their vote", fontsize=11)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("votes: healthy", votes_healthy)
c2.metric("votes: disease", votes_disease)
c3.metric("prediction", prediction)
if k == 79:
    st.info(
        "k = 79 means *the entire training set minus one* votes — the "
        "prediction barely depends on the new patient at all. It just "
        "answers 'which class is more common?'. Maximal underfitting."
    )
elif k == 1:
    st.info(
        "k = 1: the single nearest patient decides everything. Park the "
        "star next to any mislabelled-looking point deep in the wrong "
        "region and you'll inherit its label — maximal overfitting."
    )

st.header("2 · What k does to the decision boundary")
st.markdown(
    "Colour every pixel of the plane by what KNN would predict there (a "
    "**decision boundary** map — we'll draw one of these for every "
    "classifier in this section). Fitted and evaluated live for your k:"
)

clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
gx, gy = np.meshgrid(np.linspace(0, 10, 160), np.linspace(0, 10, 160))
zz = clf.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
train_acc = clf.score(X_train, y_train)
val_acc = clf.score(X_val, y_val)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
            colors=["#bbdefb", "#ffcdd2"], alpha=0.75)
for cls, colour, label in [(0, "#1565c0", "healthy"), (1, "#c62828",
                                                       "disease")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=30, label=label,
               edgecolor="white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_aspect("equal")
ax.legend(fontsize=9)
ax.set_title(f"decision regions with k = {k}", fontsize=11)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("training accuracy", f"{train_acc:.0%}")
c2.metric("validation accuracy", f"{val_acc:.0%}")
st.markdown(
    """
Sweep k from 1 upward and watch the *shape*: at k=1 the border is jagged,
with little islands wrapped around individual points (memorising — note the
perfect training accuracy), and as k grows it smooths out, until at k=79 one
class swallows the map. **k is this model's flexibility dial**, exactly like
polynomial degree on the overfitting page — small k overfits, huge k
underfits, and validation accuracy picks the sweet spot.

Two practical warnings before you use this for real:
- **Distance is scale-sensitive.** Our two biomarkers conveniently share a
  0–10 scale. Mix age (20–90) with a 0–1 lab ratio and age silently owns the
  entire distance — the fix (standardizing) gets a live demonstration on the
  Feature Engineering page.
- Prefer **odd k** for two classes, so the vote can't tie.
"""
)

st.header("3 · In code — two lines of sklearn, and no mystery underneath")
show_example(
    '''import numpy as np
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(7)
healthy = rng.normal([3.6, 4.0], 1.25, (60, 2)).clip(0, 10)
disease = rng.normal([6.4, 6.2], 1.25, (60, 2)).clip(0, 10)
X = np.vstack([healthy, disease])
y = np.array([0] * 60 + [1] * 60)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)                       # "fit" = just memorise the data
new_patient = np.array([[5.2, 4.6]])
print("prediction:", model.predict(new_patient))
print("vote shares:", model.predict_proba(new_patient))

# No mystery: reproduce that vote with the NumPy you already know
dists = np.sqrt(((X - new_patient[0]) ** 2).sum(axis=1))
nearest5 = np.argsort(dists)[:5]
print("nearest 5 labels:", y[nearest5], "-> mean:", y[nearest5].mean())''',
    """
- `model = KNeighborsClassifier(n_neighbors=5)` — every sklearn model is a class (OOP page!); hyperparameters go in when you *build* the object.
- `model.fit(X, y)` — the universal sklearn pattern. For KNN, "fitting" just stores `X` and `y`.
- `new_patient = np.array([[5.2, 4.6]])` — note the double brackets: sklearn wants a 2-D `(n_samples, n_features)` array even for one sample.
- `model.predict_proba(...)` — the class shares of the vote, e.g. `[0.8, 0.2]` = 4 of 5 neighbors said class 0.
- The last three lines re-implement the vote by hand: distances (a NumPy broadcast!), `argsort` to rank them, take 5, average the labels. Matching sklearn's output is our usual two-ways verification.
""",
)

guided_sandbox(
    key="c1",
    steps="""
1. **Step 1** — write `knn_predict(query, k)`: compute distances from
   `query` to every row of `X_train` (`np.sqrt(((X_train - query) ** 2)
   .sum(axis=1))`), take the `k` smallest with `np.argsort(dists)[:k]`, and
   return 1 if `y_train[nearest].mean() > 0.5` else 0.
2. **Step 2** — check yourself against sklearn for the patient `[5.2, 4.6]`
   with k=5 (the sklearn model is already fitted in the editor).
3. **Step 3** — loop over `k in [1, 3, 5, 9, 15, 25]`, and for each print
   the validation accuracy: `np.mean([knn_predict(x, k) == t for x, t in
   zip(X_val, y_val)])`. Which k wins?
4. **Step 4 (stretch)** — try `weights="distance"` in the sklearn model
   (closer neighbors get bigger votes) and see if it beats your best k.
""",
    setup_code='''import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
sk_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("data ready:", X_train.shape, "train,", X_val.shape, "val")

# Step 1: define knn_predict(query, k)


# Step 2: compare your prediction to sk_model.predict([[5.2, 4.6]])


# Step 3: validation accuracy for k in [1, 3, 5, 9, 15, 25]


# Step 4 (stretch): KNeighborsClassifier(n_neighbors=5, weights="distance")
''',
    solution_code='''import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
sk_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

def knn_predict(query, k):
    dists = np.sqrt(((X_train - query) ** 2).sum(axis=1))
    nearest = np.argsort(dists)[:k]
    return int(y_train[nearest].mean() > 0.5)

mine = knn_predict(np.array([5.2, 4.6]), 5)
theirs = sk_model.predict([[5.2, 4.6]])[0]
print(f"my vote: {mine}   sklearn: {theirs}   agree: {mine == theirs}")

for k in [1, 3, 5, 9, 15, 25]:
    acc = np.mean([knn_predict(x, k) == t for x, t in zip(X_val, y_val)])
    print(f"k={k:2d}: validation accuracy {acc:.0%}")

wd = KNeighborsClassifier(n_neighbors=5, weights="distance")
wd.fit(X_train, y_train)
print("weights='distance', k=5:", f"{wd.score(X_val, y_val):.0%}")''',
)
