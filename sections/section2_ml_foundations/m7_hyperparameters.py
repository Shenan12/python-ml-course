import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.sandbox import guided_sandbox, show_example

st.title("🎛️ Parameters vs Hyperparameters")
st.markdown(
    """
You've now met two very different kinds of "number in a model", and telling
them apart is essential vocabulary for reading any ML paper:

| | **Parameters** | **Hyperparameters** |
|---|---|---|
| What | the numbers *inside* the model | the settings *around* the learning |
| Examples so far | slope `w`, intercept `b`, the 12 ridge coefficients | polynomial degree, λ, learning rate, number of steps |
| Who sets them | **the algorithm**, by minimising the loss on **training** data | **you**, *before* training starts |
| How chosen | gradient descent / closed form | compared using the **validation** set |
| In DeepSurv | thousands of network weights | layers, neurons per layer, learning rate, dropout, λ… (a whole appendix of the paper) |

The giveaway question: *"could gradient descent find this number for me?"*
If yes → parameter. If it has to be fixed before the descent can even start
(how flexible is the model? how big are the steps?) → hyperparameter.

**Why can't the training loss choose hyperparameters too?** You saw the
answer on the overfitting page: training loss *always* prefers more
flexibility — it would pick degree 15 and λ=0 every time and march straight
into overfitting. Hyperparameters need the referee that punishes memorising:
validation error.
"""
)

st.header("1 · One model, both kinds of number — live")
st.markdown(
    "Same wavy mock data as the last two pages. **You** set the two "
    "hyperparameters; **the algorithm** then learns the parameters. Notice "
    "you choose *how many* parameters exist before a single one is learned."
)


def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)


x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
X_train, X_val = x_train.reshape(-1, 1), x_val.reshape(-1, 1)

DEGREES = [1, 2, 3, 5, 8, 12]
LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

c1, c2 = st.columns(2)
deg = c1.select_slider("hyperparameter 1: degree (YOU set this)", DEGREES,
                       value=3)
lam = c2.select_slider("hyperparameter 2: λ (YOU set this)", LAMBDAS,
                       value=1e-2)


def fit(deg, lam):
    m = make_pipeline(PolynomialFeatures(deg, include_bias=False),
                      StandardScaler(), Ridge(alpha=lam))
    m.fit(X_train, y_train)
    return m


model = fit(deg, lam)
coefs = model[-1].coef_
intercept = model[-1].intercept_
va_mse = np.mean((y_val - model.predict(X_val)) ** 2)

c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"""
**Hyperparameters (yours):**
- degree = `{deg}`
- λ = `{lam:g}`

**Parameters (learned just now):**
- {len(coefs)} coefficients + 1 intercept
- validation MSE: `{va_mse:.4f}`
"""
    )
with c2:
    st.code(
        "learned coefficients:\n"
        + "\n".join(f"  w{i + 1} (for x^{i + 1}) = {c: .4f}"
                    for i, c in enumerate(coefs[:6]))
        + (f"\n  … and {len(coefs) - 6} more" if len(coefs) > 6 else "")
        + f"\n  intercept  = {intercept: .4f}",
        language="text",
    )

st.header("2 · Choosing hyperparameters: the grid search")
st.markdown(
    """
Since hyperparameters can't be *learned*, they are *searched*: try every
combination, judge each on validation error, keep the winner. Below, all
**30 combinations** of degree × λ are fitted live (30 real model fits each
time this page loads). Darker green = lower validation MSE; ⭐ = the best
cell; the circle = your two sliders.
"""
)

grid_scores = np.zeros((len(DEGREES), len(LAMBDAS)))
for i, d in enumerate(DEGREES):
    for j, L in enumerate(LAMBDAS):
        m = fit(d, L)
        grid_scores[i, j] = np.mean((y_val - m.predict(X_val)) ** 2)

bi, bj = np.unravel_index(np.argmin(grid_scores), grid_scores.shape)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
im = ax.imshow(np.log10(grid_scores), cmap="RdYlGn_r", aspect="auto")
fig.colorbar(im, ax=ax, label="log10(validation MSE)")
for i in range(len(DEGREES)):
    for j in range(len(LAMBDAS)):
        ax.text(j, i, f"{grid_scores[i, j]:.3f}", ha="center", va="center",
                fontsize=8.5)
ax.scatter([bj], [bi], marker="*", s=420, facecolor="none",
           edgecolor="#1a237e", linewidth=2.5)
ax.scatter([LAMBDAS.index(lam)], [DEGREES.index(deg)], s=340,
           facecolor="none", edgecolor="#e65100", linewidth=2.5)
ax.set_xticks(range(len(LAMBDAS)))
ax.set_xticklabels([f"{L:g}" for L in LAMBDAS])
ax.set_yticks(range(len(DEGREES)))
ax.set_yticklabels(DEGREES)
ax.set_xlabel("λ")
ax.set_ylabel("degree")
ax.set_title("validation MSE for every (degree, λ) pair — "
             f"best: degree {DEGREES[bi]}, λ {LAMBDAS[bj]:g}  ⭐",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
Read the heatmap's *structure*, not just its winner:

- the **top-left region** (simple model, no fine) underfits;
- the **bottom-left** (flexible, no fine) overfits;
- the **right column** (harsh fine) underfits *regardless of degree*;
- and a **diagonal band of good cells** runs between them — more flexibility
  needs more regularization. The two hyperparameters *interact*, which is
  exactly why we search combinations rather than one knob at a time.

This humble double-loop is a real technique with a real name — **grid
search** — and it is precisely what the DeepSurv paper describes in its
appendix (they use grid search, plus a smarter cousin called random search,
over layers, learning rate, dropout and more). When you rebuild their tuning
loop in Section 6, it will be *this code with a different model inside*.

One last honesty rule, tying the whole section together: after the search
picks its winner, the winning validation score is *flattering* (you chose it
because it was lowest). The final number you'd report goes in your back
pocket: the untouched **test set**.
"""
)

st.header("3 · Grid search in code")
show_example(
    '''import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
X_train, X_val = x_train.reshape(-1, 1), x_val.reshape(-1, 1)

best = None
for degree in [1, 2, 3, 5, 8, 12]:              # hyperparameter 1
    for lam in [1e-6, 1e-4, 1e-2, 1, 100]:      # hyperparameter 2
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(), Ridge(alpha=lam))
        model.fit(X_train, y_train)             # parameters learned HERE
        score = np.mean((y_val - model.predict(X_val)) ** 2)
        if best is None or score < best[0]:
            best = (score, degree, lam)

score, degree, lam = best
print(f"winner: degree={degree}, lambda={lam:g}  (val MSE {score:.4f})")''',
    """
- The two nested `for` loops — one per hyperparameter — visit all 30 combinations. With 3 hyperparameters you'd nest three loops (and the cost multiplies: this is why smarter searches exist).
- `model.fit(X_train, y_train)` — the parameters are learned inside the loop, once per combination, on **training** data only.
- `score = ...` — each candidate judged on **validation** data. Train fits the parameters; validation picks the hyperparameters. Two piles, two jobs.
- `best = (score, degree, lam)` — the running-minimum pattern: keep the best-so-far tuple, replace when beaten. (sklearn has `GridSearchCV` to automate all this; the loop version is worth writing once so the tool is never a mystery.)
""",
)

guided_sandbox(
    key="m7",
    steps="""
1. **Step 1** — write `val_score(degree, lam)` that builds the pipeline,
   fits on the training data, and returns validation MSE (the
   `show_example` above is your reference — try from memory first).
2. **Step 2** — grid-search degrees `[1, 3, 5, 8, 12]` × lambdas
   `[1e-4, 1e-2, 1]`, printing one line per combination.
3. **Step 3** — print the winning combination, and label each number in
   your print as parameter-count or hyperparameter (e.g. "degree 5 → 5
   learned coefficients").
4. **Step 4 (stretch)** — the winner's val score is flattering (why?).
   Evaluate the winning model on the held-out `x_test, y_test` in the
   editor and compare the two numbers.
""",
    setup_code='''import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
x_test, y_test = make_data(30, seed=1234)   # untouched until step 4!
X_train = x_train.reshape(-1, 1)
X_val = x_val.reshape(-1, 1)
X_test = x_test.reshape(-1, 1)
print("data ready")

# Step 1: define val_score(degree, lam)


# Step 2: the double loop over degrees and lambdas


# Step 3: print the winner


# Step 4 (stretch): the winner's score on the TEST set
''',
    solution_code='''import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
x_test, y_test = make_data(30, seed=1234)
X_train, X_val = x_train.reshape(-1, 1), x_val.reshape(-1, 1)
X_test = x_test.reshape(-1, 1)

def build(degree, lam):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), Ridge(alpha=lam))

def val_score(degree, lam):
    model = build(degree, lam).fit(X_train, y_train)
    return np.mean((y_val - model.predict(X_val)) ** 2)

best = None
for degree in [1, 3, 5, 8, 12]:
    for lam in [1e-4, 1e-2, 1]:
        s = val_score(degree, lam)
        print(f"degree={degree:2}  lambda={lam:>6g}  val MSE={s:.4f}")
        if best is None or s < best[0]:
            best = (s, degree, lam)

s, degree, lam = best
print(f"\\nwinner: degree={degree} (hyperparameter -> {degree} learned "
      f"coefficients = parameters), lambda={lam:g} (hyperparameter)")
print(f"winning val MSE: {s:.4f}")

final = build(degree, lam).fit(X_train, y_train)
test_mse = np.mean((y_test - final.predict(X_test)) ** 2)
print(f"honest test MSE: {test_mse:.4f}  (usually a bit worse than the "
      "flattering val score - we picked the winner BECAUSE its val score "
      "was low)")''',
)
