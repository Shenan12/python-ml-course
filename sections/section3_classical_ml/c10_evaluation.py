import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, log_loss,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import cross_val_score

from utils.mockdata import patients
from utils.sandbox import guided_sandbox, show_example

st.title("📏 Model Evaluation & Comparison")
st.markdown(
    """
You can now train a dozen kinds of model — so *which is good?* "Accuracy"
alone is a trap, and this page assembles the toolkit a real ML report (and
your dissertation) actually uses: the confusion matrix, precision & recall,
the ROC curve and AUC, cross-validation, and — because you're a
statistician — **AIC** for comparing models by penalised likelihood.

Running example: our 120 patients (seed 7), a **logistic regression**
diagnosing disease. Logistic regression outputs a *probability* of disease;
a **threshold** turns that probability into a yes/no call. Almost every
metric below is really a question about *where you put that threshold*.
"""
)

X, y = patients()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
clf = LogisticRegression().fit(X_tr, y_tr)
proba = clf.predict_proba(X_va)[:, 1]

st.header("1 · The confusion matrix and the threshold")
st.markdown(
    "Slide the decision threshold and watch every patient re-sorted into "
    "four boxes. **This is the source of nearly every metric.** With a "
    "disease in mind, the two error types are not equal: a false negative "
    "(missed disease) may be far worse than a false positive (a scare and a "
    "re-test)."
)
thr = st.slider("decision threshold: call 'disease' when P(disease) ≥ …",
                0.0, 1.0, 0.5, 0.01)
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0, 1]).ravel()

c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cells = [[tn, fp], [fn, tp]]
    labels = [["TN\ntrue negative", "FP\nfalse positive\n(false alarm)"],
              ["FN\nfalse negative\n(missed!)", "TP\ntrue positive"]]
    colours = [["#c8e6c9", "#ffe0b2"], ["#ffcdd2", "#c8e6c9"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       facecolor=colours[i][j],
                                       edgecolor="#37474f"))
            ax.text(j + 0.5, 1 - i + 0.62, str(cells[i][j]), ha="center",
                    fontsize=20, fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.22, labels[i][j], ha="center",
                    fontsize=7.5)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["predicted\nhealthy", "predicted\ndisease"])
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["actually\nhealthy", "actually\ndisease"])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.tick_params(length=0)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    acc = (tp + tn) / len(y_va)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    st.metric("accuracy", f"{acc:.0%}")
    st.metric("precision  = TP/(TP+FP)", f"{prec:.0%}",
              help="of those we FLAGGED, how many truly had it")
    st.metric("recall  = TP/(TP+FN)", f"{rec:.0%}",
              help="of those who truly HAD it, how many we caught")
    st.metric("F1 (balance of the two)", f"{f1:.2f}")

st.markdown(
    """
Drag the threshold to the extremes to feel the tension:

- **threshold → 0** (flag almost everyone): recall shoots toward 100% —
  you miss no disease — but precision collapses as false alarms flood in.
- **threshold → 1** (flag almost no one): precision climbs — your rare
  flags are nearly all correct — but recall craters as real cases slip by.

This is the **precision–recall trade-off**, and it's why a single accuracy
number lies. A cancer screen rightly chooses high recall (catch everything,
tolerate false alarms); a spam filter chooses high precision (never bin a
real email). **You** pick the threshold to match which error is worse —
the model just supplies the probabilities.
"""
)

st.header("2 · ROC & AUC: judging the model at ALL thresholds at once")
st.markdown(
    """
Rather than commit to one threshold, the **ROC curve** sweeps *every*
threshold and plots, at each, the true-positive rate (recall) against the
false-positive rate. Each point is one threshold; your slider's position is
marked. The **AUC** (area under the curve) collapses the whole curve into
one number with a beautiful plain-English meaning: **the probability the
model gives a randomly chosen diseased patient a higher risk score than a
randomly chosen healthy one.** 0.5 = coin flip; 1.0 = perfect ranking.
"""
)
fpr, tpr, thresholds = roc_curve(y_va, proba)
roc_auc = auc(fpr, tpr)
cur_fpr = fp / (fp + tn) if (fp + tn) else 0
cur_tpr = tp / (tp + fn) if (tp + fn) else 0

fig, ax = plt.subplots(figsize=(6.2, 5))
ax.plot(fpr, tpr, color="#1565c0", linewidth=2.5,
        label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.fill_between(fpr, tpr, alpha=0.12, color="#1565c0")
ax.plot([0, 1], [0, 1], color="#b0bec5", linestyle="--",
        label="coin-flip model (AUC 0.5)")
ax.scatter([cur_fpr], [cur_tpr], s=140, color="#e65100", zorder=5,
           label=f"your threshold ({thr:.2f})")
ax.set_xlabel("false positive rate  =  FP/(FP+TN)")
ax.set_ylabel("true positive rate (recall)  =  TP/(TP+FN)")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("the ROC curve — your slider rides along it", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
AUC = **{roc_auc:.3f}** here. Because it never fixes a threshold, AUC is the
standard headline for comparing classifiers' *ranking* ability — and it's a
close cousin of the metric your dissertation lives on: the survival
**concordance index (C-index)** is essentially AUC generalised to censored
survival times, asking the same "did the model rank the higher-risk patient
higher?" question. Every DeepSurv and DeepHit result you'll read is a
C-index.
"""
)

st.header("3 · Cross-validation: one split is a coin toss")
st.markdown(
    """
A single train/validation split hands you *one* accuracy number — but it
depended on *which* patients happened to fall in validation. Get unlucky and
you'll misjudge the model. **k-fold cross-validation** removes the luck:
chop the data into k equal folds, then train k times, each time holding out
a different fold as the validation set. Every patient is validated exactly
once, and you get k scores to average (and to see the spread of).
"""
)
k_folds = st.slider("number of folds k", 3, 10, 5)
cv_scores = cross_val_score(LogisticRegression(), X, y, cv=k_folds)

fig, ax = plt.subplots(figsize=(9, 2.6))
fold_size = len(y) / k_folds
for i in range(k_folds):
    for j in range(k_folds):
        is_val = i == j
        ax.add_patch(plt.Rectangle((j, k_folds - 1 - i), 0.94, 0.9,
                                   facecolor="#ef6c00" if is_val else "#bbdefb",
                                   edgecolor="white"))
    ax.text(k_folds + 0.15, k_folds - 1 - i + 0.45,
            f"round {i + 1}: val acc {cv_scores[i]:.0%}", va="center",
            fontsize=9)
ax.text(-0.15, k_folds + 0.15, "orange = held-out validation fold that round",
        fontsize=8, color="#e65100")
ax.set_xlim(-0.2, k_folds + 2.6)
ax.set_ylim(0, k_folds + 0.4)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
c1, c2 = st.columns(2)
c1.metric("mean CV accuracy", f"{cv_scores.mean():.1%}")
c2.metric("std across folds (± the luck)", f"{cv_scores.std():.1%}")
st.markdown(
    f"""
Report **{cv_scores.mean():.1%} ± {cv_scores.std():.1%}**, not a lone
number — the spread tells your reader how much the estimate wobbles. This is
the honest way to compare two models: if their CV score ranges overlap
heavily, any difference may be noise. With small medical datasets (exactly
your dissertation's situation) cross-validation isn't a nicety, it's
essential — and it's why the DeepSurv paper reports cross-validated results.
"""
)

st.header("4 · AIC: comparing models by penalised likelihood")
st.markdown(
    r"""
Everything above judged models by their *predictions*. The **Akaike
Information Criterion** comes at comparison from your statistical training's
angle — **likelihood**, penalised for complexity:

$$\text{AIC} = 2k - 2\ln(\hat{L})$$

where $\ln(\hat L)$ is the fitted model's maximised log-likelihood and $k$
is its number of parameters. **Lower AIC is better.** Read it as: reward fit
(a higher likelihood lowers AIC), but charge a toll of 2 per parameter — the
exact same overfitting-defence philosophy as regularization, now expressed
in the currency of information theory rather than a validation set. It lets
you compare models on the *same data* without holding any out.

For this one demonstration we switch to the **ring dataset** from the SVM
page (diseased core inside a healthy ring), because it has a genuinely
*curved* true boundary — so there's a real decision to make about how many
polynomial features a model deserves. Each model's AIC is computed **live**
from its real log-likelihood (scikit-learn's `log_loss` is the average
negative log-likelihood, so $\ln\hat L = -n \cdot \text{log\_loss}$):
"""
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.mockdata import rings

X_aic, y_aic = rings()
max_deg = st.slider("highest polynomial degree of features to consider",
                    2, 8, 6)
rows = []
for d in range(1, max_deg + 1):
    # StandardScaler keeps the near-unpenalised (C=1e6 ≈ maximum-likelihood)
    # fit numerically well-conditioned, so the log-likelihood is trustworthy.
    m = make_pipeline(
        PolynomialFeatures(d, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=50000, C=1e6),
    ).fit(X_aic, y_aic)
    ll = -log_loss(y_aic, m.predict_proba(X_aic)[:, 1],
                   labels=[0, 1]) * len(y_aic)
    k = m[0].n_output_features_ + 1           # coefficients + intercept
    aic = 2 * k - 2 * ll
    rows.append((d, k, ll, aic))

degs = [r[0] for r in rows]
lls = [r[2] for r in rows]
aics = [r[3] for r in rows]
best_i = int(np.argmin(aics))

fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.plot(degs, aics, marker="o", color="#6a1b9a", linewidth=2, label="AIC")
ax.scatter([degs[best_i]], [aics[best_i]], s=180, marker="*", color="#2e7d32",
           zorder=5, label=f"lowest AIC at degree {degs[best_i]}")
ax2 = ax.twinx()
ax2.plot(degs, lls, marker="s", color="#0288d1", linewidth=1.5, alpha=0.6,
         label="log-likelihood")
ax.set_xlabel("polynomial degree (more features → more parameters)")
ax.set_ylabel("AIC (lower better)", color="#6a1b9a")
ax2.set_ylabel("log-likelihood (higher = better fit)", color="#0288d1")
ax.legend(fontsize=8, loc="upper center")
ax.grid(alpha=0.25)
ax.set_title("AIC = 2k − 2·log-likelihood, computed live", fontsize=10)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    "| degree | parameters k | log-likelihood | AIC |\n|---|---|---|---|\n"
    + "\n".join(
        f"| {d} | {k} | {ll:.2f} | {aic:.2f} |"
        + (" ⬅ **lowest**" if d == degs[best_i] else "")
        for (d, k, ll, aic) in rows)
)
st.markdown(
    f"""
Watch the tug-of-war in the table. Degree 1 (a straight-line boundary) fits
the ring appallingly — its log-likelihood is dreadful, so its AIC is huge.
Degree 2 adds the squared terms, which is *exactly* what a circular boundary
needs: the log-likelihood leaps, and AIC plunges to its minimum. Beyond that
the log-likelihood has nothing left to gain (the fit is already essentially
perfect) while the **`2k` toll keeps mounting** — so AIC climbs steadily
back up. The minimum sits at degree **{degs[best_i]}**, which is the honest
truth about this data: it *was* generated with a quadratic (circular)
boundary. AIC recovered the right complexity without ever seeing a
validation set.

Three things worth knowing as a statistician:
- **AIC is only comparable across models fitted to the *same* dataset.** The
  absolute value is meaningless; only differences between models matter.
- **BIC** is the close relative that swaps the penalty `2k` for
  `k·ln(n)` — a stiffer toll on large samples, so it prefers simpler models.
- **This is exactly how you'll compare survival models.** `lifelines`
  reports `AIC_` on a fitted `CoxPHFitter`, and you can compare a Cox model
  against a richer one by AIC — a direct bridge from this page to your
  dissertation. (DeepSurv itself is compared by C-index, not AIC, because
  its likelihood isn't a standard closed form — but AIC will be in your
  toolkit for the classical Cox baselines.)
"""
)

st.header("5 · In code")
show_example(
    '''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score
from utils.mockdata import patients

X, y = patients()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
clf = LogisticRegression().fit(X_tr, y_tr)
proba = clf.predict_proba(X_va)[:, 1]
pred = (proba >= 0.5).astype(int)

print("confusion matrix [[TN FP][FN TP]]:\\n", confusion_matrix(y_va, pred))
print("precision:", round(precision_score(y_va, pred), 3),
      " recall:", round(recall_score(y_va, pred), 3))
print("AUC:", round(roc_auc_score(y_va, proba), 3))

cv = cross_val_score(LogisticRegression(), X, y, cv=5)
print(f"5-fold CV accuracy: {cv.mean():.1%} +/- {cv.std():.1%}")

# AIC, by hand: 2k - 2*log-likelihood, on the full data
full = LogisticRegression().fit(X, y)
ll = -log_loss(y, full.predict_proba(X)[:, 1]) * len(y)   # log-likelihood
k = X.shape[1] + 1                                        # coefs + intercept
print("log-likelihood:", round(ll, 2), " AIC:", round(2 * k - 2 * ll, 2))''',
    """
- `confusion_matrix(y_va, pred)` — the four counts; sklearn's layout is `[[TN, FP], [FN, TP]]`, matching the coloured grid above.
- `precision_score` / `recall_score` — the two ratios directly, so you never hand-divide in real code.
- `roc_auc_score(y_va, proba)` — note it takes the **probabilities**, not the thresholded predictions: AUC is threshold-free by design.
- `cross_val_score(model, X, y, cv=5)` — hands back the 5 fold scores; `.mean()` and `.std()` give the headline ± spread. Pass a *fresh* unfitted model — it clones and refits internally per fold.
- The AIC block: `log_loss` returns the mean negative log-likelihood, so `-log_loss * n` recovers the total log-likelihood $\\ln\\hat L$; then `2k - 2·lnL` is AIC straight from the formula. `k` counts the coefficients plus the intercept.
""",
)

guided_sandbox(
    key="c10",
    steps="""
1. **Step 1** — fit `LogisticRegression()` on the training split, get
   `proba = clf.predict_proba(X_va)[:, 1]`, and print the confusion matrix
   at threshold 0.5.
2. **Step 2** — write a loop over thresholds `[0.2, 0.35, 0.5, 0.65, 0.8]`
   printing precision and recall at each (`precision_score`,
   `recall_score` on `(proba >= t)`). Watch them trade off.
3. **Step 3** — print the AUC (`roc_auc_score(y_va, proba)`) and the 5-fold
   CV accuracy mean ± std (`cross_val_score`).
4. **Step 4 (stretch)** — AIC on the **ring** data (`Xr, yr` in the editor):
   for degrees 1, 2 and 5, fit the pipeline
   `make_pipeline(PolynomialFeatures(d, include_bias=False),
   StandardScaler(), LogisticRegression(max_iter=50000, C=1e6))`, compute
   `ll = -log_loss(yr, m.predict_proba(Xr)[:, 1]) * len(yr)` and
   `k = m[0].n_output_features_ + 1`, then print `2*k - 2*ll`. Which degree
   does AIC prefer, and does it match the true shape of the boundary?
""",
    setup_code='''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils.mockdata import patients, rings

X, y = patients()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
Xr, yr = rings()          # for step 4
print("data ready:", X_tr.shape, "train,", X_va.shape, "val")

# Step 1: fit, get proba, confusion matrix at 0.5


# Step 2: precision & recall across thresholds


# Step 3: AUC and 5-fold CV mean +/- std


# Step 4 (stretch): AIC of degree 1, 2, 5 on the RING data
''',
    solution_code='''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils.mockdata import patients, rings

X, y = patients()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
Xr, yr = rings()

clf = LogisticRegression().fit(X_tr, y_tr)
proba = clf.predict_proba(X_va)[:, 1]
print("confusion at 0.5 [[TN FP][FN TP]]:\\n",
      confusion_matrix(y_va, (proba >= 0.5).astype(int)))

for t in [0.2, 0.35, 0.5, 0.65, 0.8]:
    pred = (proba >= t).astype(int)
    print(f"thr {t}: precision {precision_score(y_va, pred, zero_division=0):.2f}"
          f"  recall {recall_score(y_va, pred):.2f}")

print("AUC:", round(roc_auc_score(y_va, proba), 3))
cv = cross_val_score(LogisticRegression(), X, y, cv=5)
print(f"5-fold CV: {cv.mean():.1%} +/- {cv.std():.1%}")

def aic(degree):
    m = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                      StandardScaler(),
                      LogisticRegression(max_iter=50000, C=1e6)).fit(Xr, yr)
    ll = -log_loss(yr, m.predict_proba(Xr)[:, 1], labels=[0, 1]) * len(yr)
    k = m[0].n_output_features_ + 1
    return 2 * k - 2 * ll

for d in [1, 2, 5]:
    print(f"ring data, degree {d}: AIC {aic(d):8.2f}")
print("lowest AIC wins -> degree 2, matching the true circular boundary")''',
)
