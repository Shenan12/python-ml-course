import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.sandbox import guided_sandbox, show_example

st.title("🛠️ Feature Engineering: Scaling, Encoding, Binning")
st.markdown(
    """
Algorithms are half the job. The other half — often the *bigger* half in
practice — is preparing the features they eat. This page covers the three
preparations you'll use constantly, each with a live demonstration of why
skipping it hurts.

New mock dataset (seed 9): 220 patients with
**age** (20–90 years), an **inflammation ratio** (a lab value between 0 and
1 — and the genuinely informative feature), and **blood type** (A/B/AB/O,
a category). Disease risk rises mainly with the ratio, mildly with age.
Split: 150 train / 70 validation.
"""
)

rng = np.random.default_rng(9)
n = 220
age = rng.uniform(20, 90, n)
ratio = rng.uniform(0, 1, n)
blood = rng.choice(["A", "B", "AB", "O"], n, p=[0.42, 0.10, 0.05, 0.43])
logit = 6.0 * (ratio - 0.5) + 0.02 * (age - 55)
y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"age": age.round(1), "ratio": ratio.round(3),
                   "blood_type": blood, "disease": y})
X2 = np.column_stack([age, ratio])
X_tr, X_va = X2[:150], X2[150:]
y_tr, y_va = y[:150], y[150:]

st.dataframe(df.head(6), hide_index=True, width="stretch")

st.header("1 · Scaling: stop the loudest feature from owning the model")
st.markdown(
    """
Age spans **70 units**; the ratio spans **1**. To any distance-based model
(KNN, SVM, K-means, PCA) that's not a detail — it's everything. In the
squared distance $(\\Delta\\text{age})^2 + (\\Delta\\text{ratio})^2$, a
modest 10-year age gap contributes 100 while the ratio can contribute at
most 1: **age gets ~99% of the vote while carrying almost none of the
signal.**

Pick a treatment and watch the same KNN model (k=5), fitted live, react:
"""
)

scaler_choice = st.radio(
    "feature treatment",
    ["none (raw)", "standardize: (x − mean) / std",
     "min-max: squash into [0, 1]"],
    horizontal=True,
)
if scaler_choice.startswith("none"):
    Xt_tr, Xt_va = X_tr, X_va
elif scaler_choice.startswith("standardize"):
    sc = StandardScaler().fit(X_tr)          # statistics from TRAIN only!
    Xt_tr, Xt_va = sc.transform(X_tr), sc.transform(X_va)
else:
    sc = MinMaxScaler().fit(X_tr)
    Xt_tr, Xt_va = sc.transform(X_tr), sc.transform(X_va)

knn = KNeighborsClassifier(n_neighbors=5).fit(Xt_tr, y_tr)
acc = knn.score(Xt_va, y_va)

# distance-contribution breakdown for one illustrative pair, computed live
p1, p2 = Xt_tr[3], Xt_tr[11]
contrib = (p1 - p2) ** 2
share = contrib / contrib.sum()

c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for cls, colour, label in [(0, "#1565c0", "healthy"),
                               (1, "#c62828", "disease")]:
        m = y_tr == cls
        ax.scatter(Xt_tr[m, 0], Xt_tr[m, 1], c=colour, s=22, label=label,
                   edgecolor="white")
    ax.set_aspect("equal")
    ax.set_xlabel("age (after treatment)")
    ax.set_ylabel("ratio (after treatment)")
    ax.legend(fontsize=8)
    ax.set_title(
        "equal-units view: distances behave the way this plot LOOKS",
        fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.bar(["age", "ratio"], share * 100, color=["#7e57c2", "#26a69a"])
    ax.set_ylabel("% of squared distance")
    ax.set_ylim(0, 100)
    ax.set_title("who decides how far apart\ntwo patients are?", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    st.pyplot(fig)
    plt.close(fig)
    st.metric("KNN validation accuracy", f"{acc:.0%}")

if scaler_choice.startswith("none"):
    st.error(
        f"Raw: the cloud is a flat ribbon — vertically nothing is "
        f"'far' from anything — and age claims {share[0]:.0%} of the "
        f"example pair's distance. KNN is effectively diagnosing on age "
        f"alone: {acc:.0%}."
    )
else:
    st.success(
        f"After scaling, both features speak at comparable volume "
        f"(the pair's split is now {share[0]:.0%} / {share[1]:.0%}), the "
        f"informative ratio finally gets heard, and accuracy jumps to "
        f"{acc:.0%}. Standardize is the default choice; min-max suits "
        "bounded inputs. Trees/forests are the notable exception — they "
        "only ask threshold questions, so scaling barely moves them."
    )
st.info(
    "⚠️ The leakage rule from Section 2 in action: the scaler's mean/std "
    "were computed on the **training patients only**, then applied to "
    "validation. Scaling with all-data statistics quietly leaks the "
    "validation set into training. This page's code does it correctly — "
    "check the source in the snippet below."
)

st.header("2 · Encoding: numbers for things that aren't numbers")
st.markdown(
    """
`blood_type` is a **category** — models need numbers. The tempting shortcut
`A=0, B=1, AB=2, O=3` is a trap: it invents an *order* (O > AB > B?) and
*distances* (O is "3 times" A?) that are pure fiction, and models will
faithfully learn the fiction. The honest translation is **one-hot
encoding**: one 0/1 column per category, exactly one lit per patient.
"""
)
before = df[["age", "ratio", "blood_type"]].head(4)
after = pd.get_dummies(df[["age", "ratio", "blood_type"]],
                       columns=["blood_type"], dtype=int).head(4)
c1, c2 = st.columns(2)
c1.markdown("**before** — one text column:")
c1.dataframe(before, hide_index=True, width="stretch")
c2.markdown("**after `pd.get_dummies`** — four 0/1 columns:")
c2.dataframe(after, hide_index=True, width="stretch")
st.markdown(
    """
When *is* a plain number code fine? For **genuinely ordered** categories —
tumour stage I < II < III, education levels — where the order is real
information (that's *ordinal* encoding, and even then the "equal gaps"
assumption deserves a thought). Nominal categories like blood type: one-hot,
always. Watch out for high-cardinality traps (one-hot on "postcode" births
thousands of columns — that's when you'll meet fancier encodings).
"""
)

st.header("3 · Binning: from a continuous value to labelled buckets")
n_bins = st.slider("number of equal-width age bins", 2, 12, 5)
binned = pd.cut(df["age"], bins=n_bins)
bin_rate = df.groupby(binned, observed=True)["disease"].agg(["mean", "count"])

fig, ax = plt.subplots(figsize=(8.5, 3.6))
centers_ = [iv.mid for iv in bin_rate.index]
widths = [iv.right - iv.left for iv in bin_rate.index]
ax.bar(centers_, bin_rate["mean"], width=np.array(widths) * 0.94,
       color="#26a69a", edgecolor="#00695c", alpha=0.85)
for cx, (rate, cnt) in zip(centers_, bin_rate.values):
    ax.text(cx, rate + 0.02, f"n={int(cnt)}", ha="center", fontsize=8,
            color="#546e7a")
ax.scatter(df["age"], df["disease"] * 0.98 + 0.01, s=6, color="#37474f",
           alpha=0.25, label="individual patients (0=healthy, 1=disease)")
ax.set_xlabel("age")
ax.set_ylabel("disease rate in bin")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"pd.cut(age, bins={n_bins}): disease rate per bucket, "
             "computed live", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Binning trades detail for two gains: a **readable risk table** ("patients
70+: 62% disease rate" beats a coefficient), and letting **linear models
capture non-straight-line effects** (each bin gets its own one-hot column
and thus its own level — a poor man's curve). The cost is real: within-bin
differences vanish, and edges are arbitrary (61-year-olds and 59-year-olds
land in different buckets). Slide to 12 bins and watch small-n buckets get
jumpy — binning has its own overfitting flavour. Trees, note, effectively
*learn* their own optimal bins, which is why binning matters most for
linear-ish models. (You'll meet serious binning again when **DeepHit
discretises survival time itself** in Section 7.)
"""
)

st.header("4 · In code: a full preparation pipeline")
show_example(
    '''import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(9)
n = 220
df = pd.DataFrame({
    "age": rng.uniform(20, 90, n),
    "ratio": rng.uniform(0, 1, n),
    "blood_type": rng.choice(["A", "B", "AB", "O"], n,
                             p=[0.42, 0.10, 0.05, 0.43]),
})
logit = 6.0 * (df["ratio"] - 0.5) + 0.02 * (df["age"] - 55)
y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

X = pd.get_dummies(df, columns=["blood_type"], dtype=int)   # encode
X_tr, X_va, y_tr, y_va = X[:150], X[150:], y[:150], y[150:]

raw = KNeighborsClassifier(5).fit(X_tr, y_tr)
print("KNN on raw features:   ", round(raw.score(X_va, y_va), 3))

scaler = StandardScaler().fit(X_tr)          # statistics from train ONLY
knn = KNeighborsClassifier(5).fit(scaler.transform(X_tr), y_tr)
print("KNN on scaled features:",
      round(knn.score(scaler.transform(X_va), y_va), 3))''',
    """
- `pd.get_dummies(df, columns=["blood_type"], dtype=int)` — one-hot encodes just the named column, leaving numeric columns untouched; `dtype=int` gives clean 0/1s. (sklearn's `OneHotEncoder` does the same job inside pipelines.)
- The split comes **before** the scaler is fitted, and `StandardScaler().fit(X_tr)` sees only training rows — then the *same* fitted scaler transforms both. This ordering is the entire anti-leakage discipline in three lines.
- The two printed accuracies quantify this page's headline: same model, same information, a ~13-percentage-point gap (64% → 77% when this was last run) — from *preparation alone*.
- One-hot columns technically get standardized here too; that's common and harmless in practice, just know it's happening.
""",
)

guided_sandbox(
    key="c9",
    steps="""
1. **Step 1** — standardize `X_tr` by hand: compute `mu = X_tr.mean(axis=0)`
   and `sd = X_tr.std(axis=0)`, then `(X_tr - mu) / sd`. Print the result's
   column means (~0) and stds (~1), and check two entries against the
   `StandardScaler` output printed by the setup code.
2. **Step 2** — apply the **same** `mu` and `sd` to `X_va` (never its own!),
   fit `KNeighborsClassifier(5)` on your scaled train, and print validation
   accuracy raw vs scaled.
3. **Step 3** — one-hot the blood types: `pd.get_dummies(df["blood_type"])`
   and print `.head()`. Confirm each row has exactly one 1
   (`.sum(axis=1)`).
4. **Step 4 (stretch)** — bin age into 4 equal-width bins with
   `pd.cut(df["age"], bins=4)` and print the disease rate per bin via
   `df.groupby(bins, observed=True)["disease"].mean()`.
""",
    setup_code='''import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(9)
n = 220
df = pd.DataFrame({
    "age": rng.uniform(20, 90, n),
    "ratio": rng.uniform(0, 1, n),
    "blood_type": rng.choice(["A", "B", "AB", "O"], n,
                             p=[0.42, 0.10, 0.05, 0.43]),
})
logit = 6.0 * (df["ratio"] - 0.5) + 0.02 * (df["age"] - 55)
df["disease"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

X = df[["age", "ratio"]].to_numpy()
y = df["disease"].to_numpy()
X_tr, X_va, y_tr, y_va = X[:150], X[150:], y[:150], y[150:]
print("reference, StandardScaler on train:")
print(StandardScaler().fit_transform(X_tr)[:2].round(4))

# Step 1: standardize X_tr by hand; check means ~0, stds ~1


# Step 2: scale X_va with the TRAIN mu/sd; KNN raw vs scaled accuracy


# Step 3: one-hot blood_type; confirm one 1 per row


# Step 4 (stretch): 4 age bins and their disease rates
''',
    solution_code='''import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(9)
n = 220
df = pd.DataFrame({
    "age": rng.uniform(20, 90, n),
    "ratio": rng.uniform(0, 1, n),
    "blood_type": rng.choice(["A", "B", "AB", "O"], n,
                             p=[0.42, 0.10, 0.05, 0.43]),
})
logit = 6.0 * (df["ratio"] - 0.5) + 0.02 * (df["age"] - 55)
df["disease"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
X = df[["age", "ratio"]].to_numpy()
y = df["disease"].to_numpy()
X_tr, X_va, y_tr, y_va = X[:150], X[150:], y[:150], y[150:]

mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
Xs_tr = (X_tr - mu) / sd
print("means:", Xs_tr.mean(axis=0).round(10))
print("stds: ", Xs_tr.std(axis=0).round(10))
print("first rows:", Xs_tr[:2].round(4))

Xs_va = (X_va - mu) / sd            # train statistics, applied to val
raw = KNeighborsClassifier(5).fit(X_tr, y_tr).score(X_va, y_va)
scaled = KNeighborsClassifier(5).fit(Xs_tr, y_tr).score(Xs_va, y_va)
print(f"KNN raw: {raw:.0%}   scaled: {scaled:.0%}")

onehot = pd.get_dummies(df["blood_type"], dtype=int)
print(onehot.head())
print("every row sums to:", onehot.sum(axis=1).unique())

bins = pd.cut(df["age"], bins=4)
print(df.groupby(bins, observed=True)["disease"].mean().round(3))''',
)
