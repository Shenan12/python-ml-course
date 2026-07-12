import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

from utils.sandbox import guided_sandbox, show_example

st.title("🧭 PCA — Principal Component Analysis")
st.markdown(
    """
Real medical datasets don't have 2 features; they have 200 — and you can't
scatter-plot 200 dimensions, many features echo each other, and models
drown in the redundancy. **PCA** finds new axes for your data: the first
axis (**PC1**) points along the direction of **maximum variance** — where
the data is most spread out, i.e. most *informative* — the second (**PC2**)
is the most-varying direction perpendicular to PC1, and so on. Keep the
first few, drop the rest, and you've compressed the data while losing as
little information as possible.

Mock data (seed 42): two strongly correlated biomarkers — think "two lab
tests that largely measure the same underlying process". The cloud is a
tilted ellipse, so the interesting directions are *diagonal* — no original
axis captures them.
"""
)

r = np.random.default_rng(42)
latent = r.normal(0, 1.6, 200)
Xp = np.column_stack([5 + latent + r.normal(0, 0.55, 200),
                      5 + 0.8 * latent + r.normal(0, 0.55, 200)]).clip(0, 10)
Xc = Xp - Xp.mean(axis=0)                       # centered (PCA's first step)
pca = PCA(n_components=2).fit(Xp)

st.header("1 · Hunt the best direction yourself")
st.markdown(
    "A **projection** squashes every point onto a line — one number per "
    "patient instead of two (that's the compression). Rotate the line and "
    "watch the variance of the projected points change. PCA's claim: one "
    "specific angle preserves the most variance. Find it by hand, then "
    "check the checkbox."
)
angle = st.slider("projection direction (degrees)", 0, 179, 20)
show_pca = st.checkbox("reveal PCA's answer (computed by sklearn)")

theta = np.radians(angle)
direction = np.array([np.cos(theta), np.sin(theta)])
proj = Xc @ direction                            # real projection
proj_var = proj.var()

# variance at every angle, computed live:
angles = np.arange(0, 180)
variances = [
    (Xc @ np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])).var()
    for a in angles]
best_angle = int(angles[np.argmax(variances)])

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(Xp[:, 0], Xp[:, 1], s=18, color="#90a4ae", alpha=0.6)
    mean = Xp.mean(axis=0)
    line = np.array([mean - 5.5 * direction, mean + 5.5 * direction])
    ax.plot(line[:, 0], line[:, 1], color="#e65100", linewidth=2.2,
            label=f"your line ({angle}°)")
    # drop-lines and projected points for a subsample:
    for p in Xp[::10]:
        foot = mean + ((p - mean) @ direction) * direction
        ax.plot([p[0], foot[0]], [p[1], foot[1]], color="#ffb74d",
                linewidth=0.6, alpha=0.7)
        ax.scatter(*foot, s=14, color="#e65100", zorder=5)
    if show_pca:
        for vec, var, name, colour in [
                (pca.components_[0], pca.explained_variance_[0], "PC1",
                 "#2e7d32"),
                (pca.components_[1], pca.explained_variance_[1], "PC2",
                 "#7b1fa2")]:
            tip = mean + vec * np.sqrt(var) * 2
            ax.annotate("", xy=tip, xytext=mean,
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        linewidth=2.4))
            ax.annotate(name, tip, fontsize=11, color=colour,
                        fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("biomarker A")
    ax.set_ylabel("biomarker B")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("points squashed onto your line", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(angles, variances, color="#1565c0", linewidth=2)
    ax.scatter([angle], [proj_var], s=90, color="#e65100", zorder=5,
               label=f"your angle: variance {proj_var:.2f}")
    if show_pca:
        ax.axvline(best_angle, color="#2e7d32", linestyle=":",
                   label=f"maximum at {best_angle}° = PC1")
    ax.set_xlabel("projection angle (degrees)")
    ax.set_ylabel("variance of projected points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("variance kept, at every possible angle", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

if show_pca:
    pc1_angle = np.degrees(np.arctan2(pca.components_[0, 1],
                                      pca.components_[0, 0])) % 180
    st.success(
        f"sklearn's PC1 points at {pc1_angle:.1f}° — matching the "
        f"brute-force scan's maximum ({best_angle}°) and capturing "
        f"{pca.explained_variance_ratio_[0]:.0%} of the total variance. "
        f"PC2 (the perpendicular leftover) carries the remaining "
        f"{pca.explained_variance_ratio_[1]:.0%}. Two independent methods, "
        "one answer — that's the verification habit again."
    )
st.markdown(
    "Note the variance curve's shape: a smooth hill with its peak at PC1's "
    "angle and its valley exactly 90° away (at PC2's angle — the *least* "
    "interesting direction). PCA computes this peak directly from the "
    "data's covariance matrix (eigenvectors — you'll do it in the sandbox), "
    "no scanning required."
)

st.header("2 · Compression and its price: reconstruction")
st.markdown(
    "Keep only PC1 (one number per patient) and rebuild both biomarkers "
    "from it. The grey whiskers are the **reconstruction error** — what one "
    "component couldn't remember:"
)
pca1 = PCA(n_components=1).fit(Xp)
Xr = pca1.inverse_transform(pca1.transform(Xp))
recon_err = np.mean(((Xp - Xr) ** 2).sum(axis=1))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Xp[:, 0], Xp[:, 1], s=16, color="#90a4ae", alpha=0.5,
           label="original (2 numbers each)")
ax.scatter(Xr[:, 0], Xr[:, 1], s=16, color="#2e7d32",
           label="rebuilt from PC1 only (1 number each)")
for p, q in list(zip(Xp, Xr))[::6]:
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#b0bec5", linewidth=0.6)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.legend(fontsize=8)
ax.set_title(f"halved the storage; mean squared reconstruction error "
             f"{recon_err:.3f}", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
The rebuilt points all lie *on* the PC1 line — that's all one coordinate can
express — but they sit close to their originals, because
{pca.explained_variance_ratio_[0]:.0%} of the spread really was along that
line. This picture generalises: with 200 features, "keep enough PCs to
explain 95% of variance" might keep 12 of 200 — a 94% compression with a
princely view of the data. Uses you'll actually meet: plotting
high-dimensional data in 2-D, de-noising, and shrinking inputs for
distance-based models like KNN.

**Two rules before you use it for real:** (1) PCA is variance-hunting, so
**standardize features first** if their scales differ (next page's story) —
otherwise the biggest-scaled feature wins by cheating. (2) Variance ≠
usefulness for *your* label: PCA never looks at `y`. A direction can be
high-variance and irrelevant — it's compression, not feature selection.
"""
)

st.header("3 · In code")
show_example(
    '''import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
latent = rng.normal(0, 1.6, 200)
X = np.column_stack([5 + latent + rng.normal(0, 0.55, 200),
                     5 + 0.8 * latent + rng.normal(0, 0.55, 200)])

pca = PCA(n_components=2).fit(X)
print("components (the new axes, as unit vectors):")
print(pca.components_.round(3))
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))

# transform = express each patient in PC coordinates:
Z = pca.transform(X)
print("first patient, original:", X[0].round(2), " in PC coords:",
      Z[0].round(2))

# verify sklearn against raw linear algebra (two ways, as always):
cov = np.cov(X.T)
eigvals, eigvecs = np.linalg.eigh(cov)
print("eigenvector of the largest eigenvalue:", eigvecs[:, -1].round(3))
print("(same direction as components_[0], possibly with flipped sign)")''',
    """
- `PCA(n_components=2).fit(X)` — centers the data, then finds the principal directions. `n_components` = how many to keep (2 of 2 here, so nothing is dropped yet).
- `pca.components_` — one row per PC: the unit vector each new axis points along, expressed in original-feature coordinates.
- `explained_variance_ratio_` — each PC's share of the total variance; always sums to 1 when you keep everything. The "keep 95%" recipe reads this array.
- `pca.transform(X)` — re-expresses each patient in the new coordinates (their positions *along* PC1 and PC2). Compression = keeping only the first few columns of this.
- The last block is the statistician's verification: PCA's directions are exactly the **eigenvectors of the covariance matrix**, sorted by eigenvalue. `np.linalg.eigh` returns them ascending, so the last column pairs with `components_[0]`. A sign flip is fine — a line pointing north-east and one pointing south-west are the same line.
""",
)

guided_sandbox(
    key="c8",
    steps="""
1. **Step 1** — center the data: `Xc = X - X.mean(axis=0)`. Print the new
   column means to confirm they're ~0.
2. **Step 2** — covariance matrix `C = np.cov(Xc.T)`, then
   `eigvals, eigvecs = np.linalg.eigh(C)`. Print both.
3. **Step 3** — the largest eigenvalue is last (`eigh` sorts ascending), so
   PC1 is `eigvecs[:, -1]`. Print it next to `pca.components_[0]` from the
   setup code (sign may flip). Also verify `eigvals[-1] / eigvals.sum()`
   matches `explained_variance_ratio_[0]`.
4. **Step 4 (stretch)** — project: `z = Xc @ eigvecs[:, -1]`, print
   `z.var()` and confirm it ≈ the largest eigenvalue. You have now built
   PCA from scratch.
""",
    setup_code='''import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
latent = rng.normal(0, 1.6, 200)
X = np.column_stack([5 + latent + rng.normal(0, 0.55, 200),
                     5 + 0.8 * latent + rng.normal(0, 0.55, 200)])
pca = PCA(n_components=2).fit(X)
print("sklearn reference:")
print("  components_[0]:", pca.components_[0].round(4))
print("  explained_variance_ratio_:", pca.explained_variance_ratio_.round(4))

# Step 1: center X, confirm the column means are ~0


# Step 2: covariance matrix and its eigen-decomposition


# Step 3: compare your PC1 and variance share with sklearn's


# Step 4 (stretch): project onto PC1, check the variance = eigenvalue
''',
    solution_code='''import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
latent = rng.normal(0, 1.6, 200)
X = np.column_stack([5 + latent + rng.normal(0, 0.55, 200),
                     5 + 0.8 * latent + rng.normal(0, 0.55, 200)])
pca = PCA(n_components=2).fit(X)

Xc = X - X.mean(axis=0)
print("column means after centering:", Xc.mean(axis=0).round(10))

C = np.cov(Xc.T)
eigvals, eigvecs = np.linalg.eigh(C)
print("eigenvalues:", eigvals.round(4))
print("eigenvectors:\\n", eigvecs.round(4))

pc1_mine = eigvecs[:, -1]
print("my PC1:      ", pc1_mine.round(4))
print("sklearn PC1: ", pca.components_[0].round(4),
      "(sign flip is harmless)")
print("my variance share:     ", (eigvals[-1] / eigvals.sum()).round(4))
print("sklearn variance share:", pca.explained_variance_ratio_[0].round(4))

z = Xc @ pc1_mine
print("projection variance:", z.var(ddof=1).round(4),
      " largest eigenvalue:", eigvals[-1].round(4))''',
)
