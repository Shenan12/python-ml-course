import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans

from utils.sandbox import guided_sandbox, show_example


st.title("🧲 K-Means & Hierarchical Clustering")
st.markdown(
    """
Everything so far was **supervised**: every training patient came with a
label. Today the labels vanish. We have 150 patients' biomarkers and a
suspicion the disease has *subtypes* — but nobody has ever labelled them.
Finding groups in unlabelled data is **clustering**, the flagship of
**unsupervised learning** (this is the "data, no answers" branch promised on
the very first ML page).

**K-means** is the classic. You choose K (how many clusters to look for);
the algorithm then repeats two moves until nothing changes:

1. **Assign** — give every point to its nearest centroid (cluster centre).
2. **Update** — move each centroid to the *mean* of the points it just won.

That's the whole algorithm — no gradients, no loss surface walk, just two
moves in a loop. It's secretly minimising a quantity anyway: the total
squared distance from each point to its centroid, called **inertia**, which
each move can only decrease (assign picks nearer centroids; update picks the
within-cluster mean, the point that minimises squared distances — a fact you
know from statistics).
"""
)

r = np.random.default_rng(21)
blob_centers = np.array([[2.5, 3.0], [7.2, 7.5], [7.5, 2.2]])
X = np.vstack([r.normal(c, 0.85, (50, 2)) for c in blob_centers]).clip(0, 10)
X = X[r.permutation(len(X))]


def lloyd_history(X, k, seed):
    """Run k-means by hand, recording every (centroids, assignment) state."""
    rr = np.random.default_rng(seed)
    cents = X[rr.choice(len(X), k, replace=False)].copy()
    history = []
    for _ in range(30):
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        inertia = d[np.arange(len(X)), assign].sum()
        history.append((cents.copy(), assign.copy(), inertia))
        new_cents = np.array([X[assign == j].mean(axis=0)
                              if np.any(assign == j) else cents[j]
                              for j in range(k)])
        if np.allclose(new_cents, cents):
            break
        cents = new_cents
    return history


st.header("1 · The two-move dance, iteration by iteration")
c1, c2, c3 = st.columns(3)
k = c1.slider("K (clusters to look for)", 2, 6, 3)
seed = c2.slider("random starting centroids: seed", 0, 9, 0)
history = lloyd_history(X, k, seed)
it = c3.slider("iteration", 0, len(history) - 1, 0)

cents, assign, inertia = history[it]
PALETTE = ["#1565c0", "#c62828", "#2e7d32", "#7b1fa2", "#ef6c00", "#00838f"]

fig, ax = plt.subplots(figsize=(7.5, 6))
for j in range(k):
    m = assign == j
    ax.scatter(X[m, 0], X[m, 1], c=PALETTE[j], s=26, alpha=0.75,
               edgecolor="white")
# centroid trails: every previous position, connected
for j in range(k):
    trail = np.array([h[0][j] for h in history[:it + 1]])
    ax.plot(trail[:, 0], trail[:, 1], color=PALETTE[j], linewidth=1.6,
            alpha=0.65, zorder=4)
    ax.scatter(trail[:-1, 0], trail[:-1, 1], color=PALETTE[j], s=30,
               marker="o", alpha=0.45, zorder=4)
ax.scatter(cents[:, 0], cents[:, 1], c=PALETTE[:k], s=420, marker="X",
           edgecolor="black", linewidth=1.6, zorder=6)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_title(f"iteration {it} of {len(history) - 1}: X = current centroids, "
             "trails = where they've walked", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("inertia (total squared distance)", f"{inertia:.0f}",
          delta=None if it == 0 else f"{inertia - history[it - 1][2]:+.0f} "
          "vs previous iteration", delta_color="inverse")
c2.metric("iterations until nothing moved", len(history) - 1)
st.markdown(
    """
Drag the iteration slider like a flip-book: centroids stride across the
plane in the first two steps, points flicker allegiance at the borders, and
within a handful of iterations everything freezes — **convergence**. The
inertia readout only ever falls (the − delta), exactly as promised.

Now the two honest caveats, both visible with the sliders:

- **K-means always finds exactly K clusters — even when the data disagrees.**
  Our patients form 3 blobs: set K=2 (two blobs get welded together) or K=5
  (real blobs get carved up). The algorithm reports no discomfort either way.
- **The start matters.** With K=3, try a few seeds: most starts find the
  three blobs; an unlucky one can trap two centroids in one blob (a *local*
  minimum — inertia can't fall further but a better arrangement exists).
  Real implementations rerun from many starts and keep the best
  (sklearn's `n_init`).
"""
)

st.header("2 · Choosing K: the elbow")
inertias = [lloyd_history(X, kk, 0)[-1][2] for kk in range(1, 9)]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(range(1, 9), inertias, marker="o", color="#1565c0", linewidth=2)
ax.axvline(3, color="#2e7d32", linestyle=":", linewidth=1.6)
ax.annotate("the elbow: after K=3, extra clusters\nbuy only crumbs of "
            "inertia", (3, inertias[2]), xytext=(4.1, inertias[1]),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#2e7d32"))
ax.set_xlabel("K")
ax.set_ylabel("final inertia")
ax.grid(alpha=0.25)
ax.set_title("inertia vs K, each computed by a real k-means run", fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    "Inertia *always* falls as K grows (more centroids = shorter "
    "distances; at K=150 it would hit zero — one centroid per patient, "
    "utterly useless). So you can't just minimise it. Instead look for the "
    "**elbow**: the K where the curve stops plunging and starts crawling. "
    "Here the bend at K=3 is unmistakable — matching the 3 subtypes we "
    "secretly built into the data. Real data rarely confesses this clearly; "
    "the elbow is a judgement call, not a formula."
)

st.header("3 · Hierarchical clustering: don't choose K — build the family tree")
st.markdown(
    """
**Agglomerative (hierarchical) clustering** takes the opposite philosophy.
Start with every point as its own cluster; repeatedly **merge the two
closest clusters**; stop when one gigantic cluster remains. Recording the
merge order gives a tree — the **dendrogram** — and *every* choice of K is
just a horizontal cut through it. (Our merge rule, **Ward linkage**, merges
whichever pair increases total inertia least — the k-means-flavoured
choice. Alternatives: 'single' = closest points, 'complete' = farthest
points.)

To keep the tree readable we take 15 of the patients (5 per subtype):
"""
)

r2 = np.random.default_rng(3)
Xh = np.vstack([r2.normal(c, 0.8, (5, 2)) for c in blob_centers]).clip(0, 10)
Z = linkage(Xh, method="ward")
n_clusters = st.slider("cut the tree into how many clusters?", 1, 6, 3)
labels_h = fcluster(Z, t=n_clusters, criterion="maxclust")

# the cut height that yields n_clusters: between the right merge distances
merge_d = np.sort(Z[:, 2])
if n_clusters == 1:
    cut = merge_d[-1] * 1.05
else:
    hi = merge_d[len(merge_d) - n_clusters + 1]
    lo = merge_d[len(merge_d) - n_clusters]
    cut = (hi + lo) / 2

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    dendrogram(Z, labels=[f"p{i}" for i in range(15)], ax=ax,
               color_threshold=cut)
    ax.axhline(cut, color="#e65100", linestyle="--", linewidth=1.8)
    ax.text(0.5, cut, f" cut → {n_clusters} cluster(s)", fontsize=9,
            color="#e65100", va="bottom")
    ax.set_ylabel("merge distance (Ward)")
    ax.set_title("the dendrogram: leaves = patients,\njoins = merges, "
                 "height = how far apart", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for j in np.unique(labels_h):
        m = labels_h == j
        ax.scatter(Xh[m, 0], Xh[m, 1], c=PALETTE[(j - 1) % 6], s=90,
                   edgecolor="white")
    for i in range(15):
        ax.annotate(f"p{i}", Xh[i], textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("biomarker A")
    ax.set_ylabel("biomarker B")
    ax.set_title("the same cut, seen in feature space", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

st.markdown(
    "Slide the cut: 3 clusters slices through the three long trunks; 6 "
    "starts splitting blobs internally; 1 is the whole family. Long trunks "
    "= well-separated clusters (you must climb far before they merge) — the "
    "dendrogram's version of the elbow. Bonus: no random starts, so the "
    "answer never changes; the price is speed on big datasets (it needs "
    "all pairwise distances)."
)

st.header("4 · In code")
show_example(
    '''import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(21)
centers = np.array([[2.5, 3.0], [7.2, 7.5], [7.5, 2.2]])
X = np.vstack([rng.normal(c, 0.85, (50, 2)) for c in centers]).clip(0, 10)

km = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = km.fit_predict(X)

print("cluster sizes:", np.bincount(labels))
print("found centroids:\\n", km.cluster_centers_.round(2))
print("true blob centres:\\n", centers)
print("inertia:", round(km.inertia_, 1))
print("new patient [3, 3] joins cluster:", km.predict([[3.0, 3.0]])[0])''',
    """
- `KMeans(n_clusters=3, n_init=10, random_state=0)` — `n_init=10` runs the whole dance from 10 random starts and keeps the lowest-inertia result: the standard defence against the unlucky-start trap you saw with the seed slider.
- `fit_predict(X)` — fit, then return each point's cluster number. **Note there's no `y` anywhere** — the first fit in this course without answers.
- `np.bincount(labels)` — how many patients landed in each cluster (≈ 50/50/50 here, matching construction).
- `km.cluster_centers_` vs `centers` — the found centroids sit almost exactly on the true blob centres (mind the ordering: cluster 0 isn't necessarily blob 0 — the labels are arbitrary names).
- `km.predict([[3.0, 3.0]])` — new points can be assigned later: nearest centroid wins, same as ever.
""",
)

guided_sandbox(
    key="c7",
    steps="""
1. **Step 1** — write `assign(X, cents)`: squared distances via
   `((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)` (broadcasting —
   NumPy page!), then `.argmin(axis=1)`.
2. **Step 2** — write `update(X, labels, k)`: return
   `np.array([X[labels == j].mean(axis=0) for j in range(k)])`.
3. **Step 3** — starting from the deliberately terrible centroids in the
   editor, loop assign→update up to 20 times, printing the inertia
   (`sum of the min squared distances`) each round; stop early when the
   centroids stop changing (`np.allclose`).
4. **Step 4 (stretch)** — compare your final centroids with sklearn's
   `KMeans(n_clusters=3, n_init=10).fit(X).cluster_centers_` (order may
   differ — compare *sets* of centres by eye).
""",
    setup_code='''import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(21)
true_centers = np.array([[2.5, 3.0], [7.2, 7.5], [7.5, 2.2]])
X = np.vstack([rng.normal(c, 0.85, (50, 2))
               for c in true_centers]).clip(0, 10)
start = np.array([[1.0, 9.0], [1.5, 9.5], [2.0, 9.0]])  # awful on purpose
print("150 unlabelled patients ready; starting centroids:", start.tolist())

# Step 1: define assign(X, cents)


# Step 2: define update(X, labels, k)


# Step 3: the loop - print inertia each iteration, stop when stable


# Step 4 (stretch): compare with sklearn KMeans centers
''',
    solution_code='''import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(21)
true_centers = np.array([[2.5, 3.0], [7.2, 7.5], [7.5, 2.2]])
X = np.vstack([rng.normal(c, 0.85, (50, 2))
               for c in true_centers]).clip(0, 10)
start = np.array([[1.0, 9.0], [1.5, 9.5], [2.0, 9.0]])

def assign(X, cents):
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1), d.min(axis=1).sum()

def update(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                     else X[np.random.default_rng(j).integers(len(X))]
                     for j in range(k)])

cents = start.copy()
for i in range(20):
    labels, inertia = assign(X, cents)
    print(f"iteration {i}: inertia {inertia:8.1f}")
    new = update(X, labels, 3)
    if np.allclose(new, cents):
        print("converged.")
        break
    cents = new

print("my centroids:\\n", cents.round(2))
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("sklearn's   :\\n", km.cluster_centers_.round(2))
print("(same three centres, possibly listed in a different order)")''',
)
