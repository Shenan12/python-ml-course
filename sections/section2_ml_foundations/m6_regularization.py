import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.sandbox import guided_sandbox, show_example

st.title("🪢 L2 / Ridge Regularization")
st.markdown(
    """
Last page's overfitting antidotes were "simplify the model" and "get more
data". The third antidote is subtler and *far* more important for your
dissertation: **keep the flexible model, but punish it for using extreme
parameters**. That is regularization.

Look at what overfitting actually does to the parameters. To thread a wiggle
between every training point, a polynomial needs coefficients like
+40,000 here, −90,000 there — huge values delicately cancelling each other.
Wild coefficients ⇒ wild wiggles. So we add a fine to the loss:

$$\\text{new loss} = \\underbrace{\\text{MSE}}_{\\text{fit the data}}
\\; + \\; \\lambda \\underbrace{\\sum_j w_j^2}_{\\text{fine for big weights}}$$

The sum of *squared* weights is the **L2 penalty** ("ridge" is linear
regression + L2). The knob **λ** (lambda; called `alpha` in scikit-learn)
sets the fine's severity:

- **λ = 0** — no fine; ordinary overfitting-prone fitting.
- **λ huge** — the fine dominates; safest weights are all ≈ 0; the model
  flatlines (underfitting, bought voluntarily).
- **λ just right** — big weights now have to *pay their way*: a wiggle only
  survives if it reduces MSE by more than it costs in penalty. Noise-chasing
  wiggles can't afford that; the true pattern can.

Same data as last page, model fixed at **degree 12** — flexible enough to
overfit badly. Only λ changes below, so everything you see is the penalty's
doing.
"""
)

TRUE = lambda x: np.sin(2 * np.pi * x)  # noqa: E731


def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, TRUE(x) + r.normal(0, 0.3, n)


x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
DEGREE = 12


def fit_ridge(lam):
    reg = LinearRegression() if lam == 0 else Ridge(alpha=lam)
    model = make_pipeline(PolynomialFeatures(DEGREE, include_bias=False),
                          StandardScaler(), reg)
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


LAMBDAS = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
lam = st.select_slider("λ — the severity of the fine  (sklearn's `alpha`)",
                       LAMBDAS, value=0.0)

model = fit_ridge(lam)
grid = np.linspace(0, 1, 300).reshape(-1, 1)
tr_mse = np.mean((y_train - model.predict(x_train.reshape(-1, 1))) ** 2)
va_mse = np.mean((y_val - model.predict(x_val.reshape(-1, 1))) ** 2)
coefs = model[-1].coef_

c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(grid.ravel(), TRUE(grid.ravel()), color="#9e9e9e",
            linestyle="--", linewidth=1.6, label="true pattern")
    ax.scatter(x_train, y_train, color="#1565c0", s=30, zorder=5,
               label="train")
    ax.scatter(x_val, y_val, facecolor="none", edgecolor="#e65100", s=42,
               zorder=5, label="validation")
    ax.plot(grid.ravel(), model.predict(grid), color="#2e7d32", linewidth=2.2,
            label=f"degree-12 fit, λ={lam:g}")
    ax.set_ylim(-2.0, 2.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    ax.bar(np.arange(1, DEGREE + 1), np.abs(coefs), color="#7e57c2",
           edgecolor="#4527a0")
    ax.set_xlabel("coefficient # (x¹ … x¹²)")
    ax.set_ylabel("|coefficient| (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e5)
    ax.set_title(f"the 12 learned weights\nΣw² = {np.sum(coefs**2):.3g}",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    st.pyplot(fig)
    plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("training MSE", f"{tr_mse:.3f}")
c2.metric("validation MSE", f"{va_mse:.3f}")
c3.metric("size of the weights  Σw²", f"{np.sum(coefs ** 2):.3g}")
st.markdown(
    "Slide λ from 0 upward and watch the *mechanism*: the weight bars sink "
    "(right chart), the wiggles die (left chart), the validation MSE falls — "
    "then keep pushing to λ=100 and watch the model flatline into "
    "underfitting. Same slider, both failure modes."
)

st.header("The U-curve again — but now λ is the dial")
tr_curve, va_curve, w2 = [], [], []
for L in LAMBDAS:
    m = fit_ridge(L)
    tr_curve.append(np.mean((y_train - m.predict(x_train.reshape(-1, 1))) ** 2))
    va_curve.append(np.mean((y_val - m.predict(x_val.reshape(-1, 1))) ** 2))
    w2.append(np.sum(m[-1].coef_ ** 2))
best_lam = LAMBDAS[int(np.argmin(va_curve))]

fig, ax = plt.subplots(figsize=(8.5, 3.8))
xticks = np.arange(len(LAMBDAS))
ax.semilogy(xticks, tr_curve, marker="o", color="#1565c0", label="train MSE")
ax.semilogy(xticks, va_curve, marker="s", color="#e65100", label="val MSE")
ax.set_xticks(xticks)
ax.set_xticklabels([f"{L:g}" for L in LAMBDAS])
ax.axvline(LAMBDAS.index(best_lam), color="#2e7d32", linestyle=":",
           label=f"best λ = {best_lam:g}")
ax.scatter([LAMBDAS.index(lam)], [va_curve[LAMBDAS.index(lam)]], s=140,
           facecolor="none", edgecolor="#2e7d32", linewidth=2, zorder=6,
           label="your slider")
ax.set_xlabel("λ  (weak fine → harsh fine)")
ax.set_ylabel("MSE (log scale)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
Mirror-image of the degree U-curve: **left = overfitting** (no fine),
**right = underfitting** (crushing fine), sweet spot at λ = {best_lam:g} for
this data — chosen, as ever, by the validation set. λ is our second
*hyperparameter* (after degree), a word the next page pins down properly.

**Why this matters for the papers you'll read:** the DeepSurv loss function
is *exactly* this pattern — its negative log partial likelihood plus an L2
penalty on the network's weights. When you meet
$\\lambda \\cdot \\|\\theta\\|^2_2$ in Section 6, you have already used it.
(You may also meet **L1/lasso** — fining $\\sum |w_j|$ instead, which pushes
weights to exactly zero — worth knowing exists, not needed further here.)
"""
)

st.header("In code: three lines change")
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

print("   lambda | val MSE | size of weights sum(w^2)")
for lam in [1e-6, 1e-4, 1e-2, 1, 100]:
    model = make_pipeline(PolynomialFeatures(12, include_bias=False),
                          StandardScaler(),
                          Ridge(alpha=lam))
    model.fit(x_train.reshape(-1, 1), y_train)
    val_mse = np.mean((y_val - model.predict(x_val.reshape(-1, 1))) ** 2)
    w2 = np.sum(model[-1].coef_ ** 2)
    print(f"{lam:9g} | {val_mse:7.4f} | {w2:.3g}")''',
    """
- `PolynomialFeatures(12, include_bias=False)` — expands the single feature `x` into 12 features `x, x², …, x¹²`; a degree-12 polynomial is then just a *linear* model on those columns.
- `StandardScaler()` — standardizes each feature (subtract its training mean, divide by its training std — the exact NumPy-page computation). Essential here: `x¹²` lives on a microscopic scale next to `x`, and the penalty `Σw²` is only fair if features share a scale. DeepSurv standardizes its inputs for the same family of reasons.
- `make_pipeline(...)` — chains steps so they run as one model; `.fit` runs them in order (and the scaler learns its means from *training data only* — the leakage rule from the splits page, handled correctly for free).
- `Ridge(alpha=lam)` — linear regression whose loss is MSE + `alpha`·Σw². That one argument is the entire implementation of this page.
- `x_train.reshape(-1, 1)` — sklearn wants `X` as 2-D `(n_samples, n_features)`; reshape turns the flat 25-vector into a 25×1 column (`-1` = "work this dimension out yourself").
- `model[-1].coef_` — the learned weights of the final pipeline step. The printed `sum(w^2)` column collapsing as λ grows *is* the penalty at work.
""",
)

guided_sandbox(
    key="m6",
    steps="""
1. **Step 1** — write `val_mse_for(lam)`: build the
   `make_pipeline(PolynomialFeatures(12, include_bias=False),
   StandardScaler(), Ridge(alpha=lam))` model, fit it on the training data,
   and return the validation MSE.
2. **Step 2** — loop over `lambdas` (given in the editor), print each λ and
   its validation MSE, and print the best λ.
3. **Step 3** — for the best λ and for λ=1e-6, print
   `np.sum(model[-1].coef_ ** 2)`. How many times smaller are the winning
   weights? (You may need `val_mse_for` to also return the model, or just
   rebuild it.)
4. **Step 4 (stretch)** — swap `Ridge` for `LinearRegression()` (import is
   already there) and check it behaves like the λ→0 end of your table.
""",
    setup_code='''import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
X_train = x_train.reshape(-1, 1)
X_val = x_val.reshape(-1, 1)

lambdas = [1e-6, 1e-4, 1e-2, 1, 100]
print("data ready")

# Step 1: define val_mse_for(lam)


# Step 2: loop over lambdas, print val MSEs and the best lambda


# Step 3: compare sum(coef**2) at the best lambda vs at 1e-6


# Step 4 (stretch): LinearRegression() as the lambda -> 0 limit
''',
    solution_code='''import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
X_train = x_train.reshape(-1, 1)
X_val = x_val.reshape(-1, 1)
lambdas = [1e-6, 1e-4, 1e-2, 1, 100]

def build(lam):
    return make_pipeline(PolynomialFeatures(12, include_bias=False),
                         StandardScaler(), Ridge(alpha=lam))

def val_mse_for(lam):
    model = build(lam).fit(X_train, y_train)
    return np.mean((y_val - model.predict(X_val)) ** 2)

scores = {lam: val_mse_for(lam) for lam in lambdas}
for lam, s in scores.items():
    print(f"lambda {lam:>6g}: val MSE {s:.4f}")
best = min(scores, key=scores.get)
print("best lambda:", best)

w2_best = np.sum(build(best).fit(X_train, y_train)[-1].coef_ ** 2)
w2_tiny = np.sum(build(1e-6).fit(X_train, y_train)[-1].coef_ ** 2)
print(f"sum(w^2): best={w2_best:.3g}  vs  lam=1e-6: {w2_tiny:.3g} "
      f"({w2_tiny / w2_best:.0f}x bigger)")

lin = make_pipeline(PolynomialFeatures(12, include_bias=False),
                    StandardScaler(), LinearRegression()).fit(X_train, y_train)
lin_mse = np.mean((y_val - lin.predict(X_val)) ** 2)
print(f"plain LinearRegression val MSE: {lin_mse:.4f} "
      "(the no-fine extreme)")''',
)
