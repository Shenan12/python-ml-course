import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.svm import SVC

from utils.mockdata import patients_split, rings
from utils.sandbox import guided_sandbox, show_example

st.title("🛣️ Support Vector Machines (SVM)")
st.markdown(
    """
KNN and trees draw their boundaries as by-products. The SVM asks the
boundary question head-on: **of all the lines that separate the classes,
which is THE best one?** Its answer: the line with the widest possible
safety corridor — the **margin** — between itself and the nearest points of
each class. A wide margin means new patients near the border still tend to
land on the correct side.

Three words to own before the pictures:

- **Margin** — the corridor between the boundary and the nearest training
  points. The SVM *maximises its width*.
- **Support vectors** — the handful of points sitting on (or inside) the
  corridor. **They alone determine the boundary** — every other point could
  be deleted without moving the line an inch. (That's where the name comes
  from: these vectors *support* the boundary.)
- **C** — the rule-bending hyperparameter. Real data overlaps, so a rigid
  corridor may be impossible or absurdly thin. Small C = relaxed: allow some
  points inside the corridor (even misclassified) in exchange for a wide,
  stable margin. Huge C = strict: violations are punished so hard the
  margin shrinks to appease individual points — hello, overfitting.
"""
)

X_train, X_val, y_train, y_val = patients_split()

st.header("1 · The margin, the corridor, and C — live")
C = st.select_slider("C (margin strictness)", [0.01, 0.1, 1.0, 10.0, 1000.0],
                     value=1.0)
svm = SVC(kernel="linear", C=C).fit(X_train, y_train)

gx, gy = np.meshgrid(np.linspace(0, 10, 220), np.linspace(0, 10, 220))
grid = np.column_stack([gx.ravel(), gy.ravel()])
dec = svm.decision_function(grid).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contourf(gx, gy, dec, levels=[-1e9, 0, 1e9], colors=["#bbdefb", "#ffcdd2"],
            alpha=0.6)
ax.contour(gx, gy, dec, levels=[-1, 0, 1], colors="#37474f",
           linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
for cls, colour, label in [(0, "#1565c0", "healthy"),
                           (1, "#c62828", "disease")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=30, label=label,
               edgecolor="white", zorder=3)
sv = svm.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=160, facecolor="none", edgecolor="#2e7d32",
           linewidth=1.8, zorder=4,
           label=f"support vectors ({len(sv)})")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"solid = boundary, dashed = margin edges  (C = {C:g})",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("support vectors", len(sv), f"of {len(X_train)} training points")
c2.metric("training accuracy", f"{svm.score(X_train, y_train):.0%}")
c3.metric("validation accuracy", f"{svm.score(X_val, y_val):.0%}")
st.markdown(
    """
Sweep C and watch the *mechanism*: at C = 0.01 the corridor is enormous and
half the dataset is circled (all inside the relaxed margin, all with a say);
at C = 1000 only a hard core of borderline patients remains circled and the
corridor squeezes thin around them. The green circles are the entire model —
sklearn stores only those points.
"""
)

st.header("2 · The kernel trick: when no straight line can work")
st.markdown(
    """
New mock disease (seed 12): patients are sick when **both biomarkers are
mid-range** — the diseased core sits *inside* a healthy ring. No straight
line can ever separate a ring from its centre.

The SVM's famous escape is to **add a dimension**. Give every patient a
third, manufactured coordinate — its squared distance from the centre,
$z = (A-5)^2 + (B-5)^2$ — and look at the data side-on: the core sinks, the
ring floats, and a *flat plane* separates them perfectly. A curved boundary
in 2-D is just the shadow of a straight one in 3-D.
"""
)

Xr, yr = rings()
z = (Xr[:, 0] - 5) ** 2 + (Xr[:, 1] - 5) ** 2

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for cls, colour, label in [(0, "#1565c0", "healthy (ring)"),
                               (1, "#c62828", "disease (core)")]:
        m = yr == cls
        ax.scatter(Xr[m, 0], Xr[m, 1], c=colour, s=26, label=label,
                   edgecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("biomarker A")
    ax.set_ylabel("biomarker B")
    ax.set_title("original 2-D view: hopeless for a line", fontsize=10)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
        m = yr == cls
        ax.scatter(Xr[m, 0], z[m], c=colour, s=26, edgecolor="white")
    thresh_z = (z[yr == 0].min() + z[yr == 1].max()) / 2
    ax.axhline(thresh_z, color="#2e7d32", linewidth=2.2,
               label=f"a flat cut at z = {thresh_z:.1f}")
    ax.set_xlabel("biomarker A")
    ax.set_ylabel("manufactured feature  z = (A−5)² + (B−5)²")
    ax.set_title("same patients, seen from the side in 3-D", fontsize=10)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

st.markdown(
    """
Here we hand-crafted the extra dimension because we secretly knew the
disease was "distance from the centre". The **kernel trick** is the general,
knowledge-free version: a kernel (we use **RBF**, the default) lets the SVM
behave *as if* the data were lifted into an enormously high-dimensional
space — without ever computing the coordinates, only similarities between
pairs of points. Below, both attempts run live on the ring data:
"""
)
kernel = st.radio("kernel", ["linear (doomed)", "rbf (the trick)"],
                  horizontal=True, index=1)
gamma = st.select_slider("gamma — how *local* the rbf similarity is",
                         [0.01, 0.1, 1.0, 10.0], value=1.0,
                         disabled=kernel.startswith("linear"))

ksvm = (SVC(kernel="linear", C=1.0) if kernel.startswith("linear")
        else SVC(kernel="rbf", C=1.0, gamma=gamma)).fit(Xr, yr)
zz = ksvm.predict(grid).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#bbdefb", "#ffcdd2"],
            alpha=0.75)
for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
    m = yr == cls
    ax.scatter(Xr[m, 0], Xr[m, 1], c=colour, s=24, edgecolor="white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
acc = ksvm.score(Xr, yr)
ax.set_title(f"{kernel.split()[0]} kernel — accuracy {acc:.0%}", fontsize=10)
st.pyplot(fig)
plt.close(fig)
if kernel.startswith("linear"):
    st.error(
        f"As promised: the best straight line manages {acc:.0%} — barely "
        "better than guessing, because every line has ring on both sides."
    )
else:
    st.success(
        f"The RBF kernel wraps a smooth boundary around the core: "
        f"{acc:.0%}. Try gamma = 0.01 (so global it blurs back toward a "
        "line) and gamma = 10 (so local it shrink-wraps noise — "
        "overfitting's newest costume)."
    )

st.header("3 · In code")
show_example(
    '''from sklearn.svm import SVC
from utils.mockdata import patients_split, rings

X_train, X_val, y_train, y_val = patients_split()

linear = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
print("linear SVM on the blobs:")
print("  support vectors:", linear.support_vectors_.shape[0],
      "of", X_train.shape[0], "training points")
print("  validation accuracy:", linear.score(X_val, y_val))

Xr, yr = rings()
for kernel in ["linear", "rbf"]:
    model = SVC(kernel=kernel, C=1.0).fit(Xr, yr)
    print(f"{kernel:6} kernel on the RING data: accuracy "
          f"{model.score(Xr, yr):.0%}")''',
    """
- `SVC(kernel="linear", C=1.0)` — Support Vector Classifier; `kernel=` chooses the geometry (straight vs lifted), `C` the strictness. Same `.fit`/`.score` ritual as every sklearn model — by the third algorithm the interface is muscle memory, which is precisely sklearn's design.
- `linear.support_vectors_` — the stored borderline points; the trailing underscore is sklearn's convention for *"learned during fit"*.
- The loop makes the ring comparison honest: same data, same C, only the kernel changes — the accuracy gap is the kernel trick, isolated.
- One practical note for real use: SVMs are distance-based like KNN, so **standardize features first** (Feature Engineering page) — our biomarkers just happen to share a scale already.
""",
)

guided_sandbox(
    key="c3",
    steps="""
1. **Step 1** — fit `SVC(kernel="linear", C=1.0)` on the blob training data
   and print its validation accuracy and its number of support vectors
   (`len(model.support_vectors_)`).
2. **Step 2** — loop `C in [0.01, 0.1, 1, 10, 1000]` and print support
   vector count + validation accuracy for each. Describe the trend you see
   in the counts.
3. **Step 3** — on the ring data `Xr, yr`, build the manufactured feature
   `z = (Xr[:, 0] - 5)**2 + (Xr[:, 1] - 5)**2`, stack it with
   `np.column_stack([Xr, z])`, and fit a **linear** SVC on this 3-feature
   version. Print its accuracy — you've done the kernel trick by hand.
4. **Step 4 (stretch)** — compare with `SVC(kernel="rbf")` on the plain 2-D
   ring data, and try gamma 0.01 / 1 / 10.
""",
    setup_code='''import numpy as np
from sklearn.svm import SVC
from utils.mockdata import patients_split, rings

X_train, X_val, y_train, y_val = patients_split()
Xr, yr = rings()
print("blobs:", X_train.shape, "  rings:", Xr.shape)

# Step 1: linear SVC on the blobs - val accuracy + support vector count


# Step 2: loop over C values


# Step 3: hand-made kernel trick on the rings (add z, fit linear SVC)


# Step 4 (stretch): rbf kernel on the plain rings, various gamma
''',
    solution_code='''import numpy as np
from sklearn.svm import SVC
from utils.mockdata import patients_split, rings

X_train, X_val, y_train, y_val = patients_split()
Xr, yr = rings()

model = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
print(f"C=1: val acc {model.score(X_val, y_val):.0%}, "
      f"{len(model.support_vectors_)} support vectors")

for C in [0.01, 0.1, 1, 10, 1000]:
    m = SVC(kernel="linear", C=C).fit(X_train, y_train)
    print(f"C={C:>6}: {len(m.support_vectors_):3d} support vectors, "
          f"val acc {m.score(X_val, y_val):.0%}")
print("-> bigger C = stricter = fewer points allowed near the boundary")

z = (Xr[:, 0] - 5) ** 2 + (Xr[:, 1] - 5) ** 2
X3 = np.column_stack([Xr, z])
lifted = SVC(kernel="linear", C=1.0).fit(X3, yr)
print(f"hand-lifted 3-feature linear SVM on rings: "
      f"{lifted.score(X3, yr):.0%}")

for g in [0.01, 1, 10]:
    rbf = SVC(kernel="rbf", C=1.0, gamma=g).fit(Xr, yr)
    print(f"rbf gamma={g:>5}: accuracy {rbf.score(Xr, yr):.0%}")''',
)
