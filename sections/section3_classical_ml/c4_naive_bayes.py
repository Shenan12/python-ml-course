import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.naive_bayes import GaussianNB

from utils.mockdata import patients_split
from utils.sandbox import guided_sandbox, show_example

st.title("🎲 Naive Bayes")
st.markdown(
    r"""
As a statistics student you arrive at this page with home advantage: Naive
Bayes is just **Bayes' theorem used as a classifier**.

$$P(\text{disease} \mid \text{biomarkers}) \;=\;
\frac{P(\text{biomarkers} \mid \text{disease}) \; P(\text{disease})}
     {P(\text{biomarkers})}$$

Read as a diagnosis recipe: start from the **prior** (how common is the
disease among training patients?), multiply by the **likelihood** (how
typical are *these* biomarker values among diseased patients?), normalise,
and you have the **posterior** — the probability this patient is diseased.
Compute it for both classes; predict the larger.

The catch: the likelihood $P(A, B \mid \text{class})$ is a *joint*
distribution, and with many features it's hopeless to estimate. The
**"naive" step** assumes features are independent within each class:

$$P(A, B \mid \text{class}) \;\approx\;
P(A \mid \text{class}) \cdot P(B \mid \text{class})$$

— so we only need each feature's distribution *separately*. **Gaussian**
Naive Bayes models each of those as a normal distribution, fitted with
nothing more than each feature's per-class mean and variance. The assumption
is usually false (our biomarkers are correlated!) yet the classifier often
works anyway — the ranking of posteriors survives more abuse than their
exact values.
"""
)

X_train, X_val, y_train, y_val = patients_split()

# Per-class fitted Gaussians (the entire "model"):
stats = {}
for cls in [0, 1]:
    Xc = X_train[y_train == cls]
    stats[cls] = {"mean": Xc.mean(axis=0), "var": Xc.var(axis=0),
                  "prior": (y_train == cls).mean()}


def normal_pdf(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


st.header("1 · Watch the posterior get built, factor by factor")
st.markdown(
    "The bell curves are the fitted per-class normals for each biomarker "
    "(their means/variances computed live from the 80 training patients). "
    "Move the new patient and watch each factor get read off the curves, "
    "multiplied, and normalised:"
)
c1, c2 = st.columns(2)
qa = c1.slider("new patient: biomarker A", 0.0, 10.0, 4.4, 0.1)
qb = c2.slider("new patient: biomarker B", 0.0, 10.0, 6.0, 0.1)

xs = np.linspace(0, 10, 300)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for j, (ax, fname, q) in enumerate(zip(axes, ["biomarker A", "biomarker B"],
                                       [qa, qb])):
    for cls, colour, label in [(0, "#1565c0", "healthy"),
                               (1, "#c62828", "disease")]:
        pdf = normal_pdf(xs, stats[cls]["mean"][j], stats[cls]["var"][j])
        ax.plot(xs, pdf, color=colour, linewidth=2, label=label)
        lik = normal_pdf(q, stats[cls]["mean"][j], stats[cls]["var"][j])
        ax.scatter([q], [lik], color=colour, s=70, zorder=5)
        ax.annotate(f"{lik:.3f}", (q, lik), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color=colour)
    ax.axvline(q, color="#616161", linestyle=":", linewidth=1.2)
    ax.set_xlabel(fname)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
fig.suptitle("likelihood factors: read each class's curve at the patient's "
             "value", fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# The naive computation, by hand, live:
rows = {}
for cls in [0, 1]:
    lik_a = normal_pdf(qa, stats[cls]["mean"][0], stats[cls]["var"][0])
    lik_b = normal_pdf(qb, stats[cls]["mean"][1], stats[cls]["var"][1])
    rows[cls] = (stats[cls]["prior"], lik_a, lik_b,
                 stats[cls]["prior"] * lik_a * lik_b)
total = rows[0][3] + rows[1][3]
post = {cls: rows[cls][3] / total for cls in [0, 1]}

st.markdown(
    f"""
| | prior | × P(A={qa:.1f} \\| class) | × P(B={qb:.1f} \\| class) | = unnormalised | → posterior |
|---|---|---|---|---|---|
| **healthy** | {rows[0][0]:.2f} | {rows[0][1]:.4f} | {rows[0][2]:.4f} | {rows[0][3]:.6f} | **{post[0]:.1%}** |
| **disease** | {rows[1][0]:.2f} | {rows[1][1]:.4f} | {rows[1][2]:.4f} | {rows[1][3]:.6f} | **{post[1]:.1%}** |
"""
)

nb = GaussianNB().fit(X_train, y_train)
sk_post = nb.predict_proba([[qa, qb]])[0]
verdict = "disease" if post[1] > post[0] else "healthy"
st.success(
    f"Our hand computation says **{verdict}** with P(disease) = "
    f"{post[1]:.1%}. sklearn's `GaussianNB.predict_proba` for the same "
    f"patient: {sk_post[1]:.1%} — matching to rounding, because the table "
    "above *is* the entire algorithm."
)

st.header("2 · The decision boundary it produces")
gx, gy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
grid = np.column_stack([gx.ravel(), gy.ravel()])
proba = nb.predict_proba(grid)[:, 1].reshape(gx.shape)

fig, ax = plt.subplots(figsize=(7, 5.4))
cs = ax.contourf(gx, gy, proba, levels=np.linspace(0, 1, 11), cmap="RdBu_r",
                 alpha=0.8)
fig.colorbar(cs, ax=ax, label="P(disease | A, B)")
ax.contour(gx, gy, proba, levels=[0.5], colors="black", linewidths=2)
for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour, s=24,
               edgecolor="white")
ax.scatter([qa], [qb], marker="*", s=320, c="#ffd600", edgecolor="black",
           zorder=6)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
ax.set_title("Naive Bayes outputs a full probability surface — "
             "the black line is the 50/50 frontier", fontsize=10)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("training accuracy", f"{nb.score(X_train, y_train):.0%}")
c2.metric("validation accuracy", f"{nb.score(X_val, y_val):.0%}")
st.markdown(
    """
Notice what you get that KNN and trees didn't emphasise: **graded
probabilities everywhere**, not just a verdict — deep blue is "confidently
healthy", the pale band is "genuinely unsure". Probability outputs like this
feed straight into the threshold/ROC machinery on the Model Evaluation page.

Where Naive Bayes shines in practice: **text** (the classic spam filter —
thousands of word-count features, where the independence approximation is
tolerable and speed is unbeatable), tiny training sets, and as an honest
baseline before anything fancy. Its weakness: correlated features get
double-counted — it becomes overconfident (posteriors like 99.99%) even when
its *accuracy* holds up.
"""
)

st.header("3 · In code")
show_example(
    '''import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()

nb = GaussianNB().fit(X_train, y_train)
print("validation accuracy:", nb.score(X_val, y_val))

# The ENTIRE fitted model is just these numbers:
print("per-class feature means:\\n", nb.theta_.round(2))
print("per-class feature variances:\\n", nb.var_.round(2))
print("priors:", nb.class_prior_)

patient = np.array([[4.4, 6.0]])
print("P(healthy), P(disease):", nb.predict_proba(patient).round(3))''',
    """
- `GaussianNB()` — no hyperparameters needed for the basic version; fitting only computes means, variances and priors. Trains effectively instantly, on any amount of data.
- `nb.theta_` and `nb.var_` — a 2×2 array each (2 classes × 2 features): the bell curves from the visualization, and *the whole model*. Compare the printed means with where the curves peak above.
- `nb.class_prior_` — the class frequencies in training data ([0.5, 0.5] here since our mock data is balanced — with a rare disease the prior would do real work, dragging every posterior toward "healthy").
- `predict_proba(...)` — the normalised posteriors; `predict` just returns the argmax of these.
""",
)

guided_sandbox(
    key="c4",
    steps="""
1. **Step 1** — for each class, compute the per-feature `mean` and `var` of
   `X_train[y_train == cls]` (both with `axis=0`) and the prior
   `(y_train == cls).mean()`. Print them; check against `nb.theta_` /
   `nb.var_` printed by the setup code.
2. **Step 2** — write `normal_pdf(x, mean, var)` using the formula
   `np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)`.
3. **Step 3** — for the patient `[4.4, 6.0]`, compute each class's
   `prior * normal_pdf(A) * normal_pdf(B)`, normalise the two, and print
   P(disease). Compare with `nb.predict_proba` from the setup code.
4. **Step 4 (stretch)** — your posteriors multiply three factors; for 30
   features you'd multiply 31 tiny numbers and *underflow* to 0.0. Redo
   step 3 with `np.log`: sum `np.log(prior) + np.log(pdf_A) + np.log(pdf_B)`
   per class and predict via the larger sum. (This log-sum is how every real
   implementation works, and log-likelihoods return in DeepSurv's loss.)
""",
    setup_code='''import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
nb = GaussianNB().fit(X_train, y_train)
print("sklearn reference:")
print("  theta_ (means):\\n", nb.theta_.round(3))
print("  var_:\\n", nb.var_.round(3))
print("  predict_proba([[4.4, 6.0]]):", nb.predict_proba([[4.4, 6.0]]).round(4))

# Step 1: per-class means, variances, priors - print and compare


# Step 2: define normal_pdf(x, mean, var)


# Step 3: posterior for patient [4.4, 6.0], compare with sklearn


# Step 4 (stretch): the same decision using log-probabilities
''',
    solution_code='''import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()
nb = GaussianNB().fit(X_train, y_train)

stats = {}
for cls in [0, 1]:
    Xc = X_train[y_train == cls]
    stats[cls] = (Xc.mean(axis=0), Xc.var(axis=0), (y_train == cls).mean())
    print(f"class {cls}: mean {stats[cls][0].round(3)}, "
          f"var {stats[cls][1].round(3)}, prior {stats[cls][2]}")

def normal_pdf(x, mean, var):
    return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

patient = np.array([4.4, 6.0])
unnorm = {}
for cls in [0, 1]:
    mean, var, prior = stats[cls]
    unnorm[cls] = (prior * normal_pdf(patient[0], mean[0], var[0])
                   * normal_pdf(patient[1], mean[1], var[1]))
p_disease = unnorm[1] / (unnorm[0] + unnorm[1])
print(f"my P(disease) = {p_disease:.4f}")
print("sklearn       =", nb.predict_proba([patient])[0][1].round(4))

log_scores = {}
for cls in [0, 1]:
    mean, var, prior = stats[cls]
    log_scores[cls] = (np.log(prior)
                       + np.log(normal_pdf(patient[0], mean[0], var[0]))
                       + np.log(normal_pdf(patient[1], mean[1], var[1])))
print("log-scores:", {c: round(s, 3) for c, s in log_scores.items()},
      "-> predict class", max(log_scores, key=log_scores.get))''',
)
