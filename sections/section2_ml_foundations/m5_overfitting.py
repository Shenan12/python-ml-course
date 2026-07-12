import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.sandbox import guided_sandbox, show_example

st.title("🎭 Overfitting vs Underfitting")
st.markdown(
    """
Now the splits earn their keep. We'll fit models of increasing flexibility
and watch the **training** score and the **validation** score tell two
different stories. This tension has a name on each side:

- **Underfitting** — the model is too simple to capture the pattern. It's
  bad on the training data *and* bad on new data. (A flat line through a
  curve.)
- **Overfitting** — the model is so flexible it fits the training points'
  *noise*, not just their pattern. Spectacular on training data, terrible on
  new data. (It memorised the homework.)

Our mock data (seed 5): a smooth wavy **true pattern** — we know it exactly
because we generated it — plus random noise, split into training and
validation points. The model: a polynomial of degree *d* (degree 1 = a
line, degree 3 = a cubic, degree 15 = a 16-parameter wiggle-monster). The
degree is your flexibility dial.
"""
)

TRUE = lambda x: np.sin(2 * np.pi * x)  # noqa: E731
rng = np.random.default_rng(5)
NOISE = 0.30


def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, TRUE(x) + r.normal(0, NOISE, n)


x_val, y_val = make_data(30, seed=99)


def fit_poly(x, y, degree):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.polyfit(x, y, degree)


st.header("1 · The flexibility dial")
c1, c2 = st.columns(2)
degree = c1.slider("polynomial degree d", 0, 15, 1)
n_train = c2.select_slider("number of training points", [15, 25, 100],
                           value=25)
x_train, y_train = make_data(n_train, seed=5)

coeffs = fit_poly(x_train, y_train, degree)
grid = np.linspace(0, 1, 300)
train_mse = np.mean((y_train - np.polyval(coeffs, x_train)) ** 2)
val_mse = np.mean((y_val - np.polyval(coeffs, x_val)) ** 2)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(grid, TRUE(grid), color="#9e9e9e", linestyle="--", linewidth=1.8,
        label="the TRUE pattern (secret in real life)")
ax.scatter(x_train, y_train, color="#1565c0", s=32, zorder=5,
           label=f"training points ({n_train})")
ax.scatter(x_val, y_val, facecolor="none", edgecolor="#e65100", s=45,
           zorder=5, label="validation points (30)")
ax.plot(grid, np.polyval(coeffs, grid), color="#2e7d32", linewidth=2.2,
        label=f"fitted degree-{degree} polynomial")
ax.set_ylim(-2.0, 2.0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("training MSE", f"{train_mse:.3f}")
c2.metric("validation MSE", f"{val_mse:.3f}",
          delta=f"{val_mse - train_mse:+.3f} vs train", delta_color="inverse")

if degree <= 1:
    st.warning(
        f"**Underfitting.** A degree-{degree} model can't bend enough to "
        f"follow the wave — both scores are poor (train {train_mse:.3f}, "
        f"val {val_mse:.3f}). No amount of extra data fixes a model that's "
        "too simple: try it with the data slider."
    )
elif degree <= 6:
    st.success(
        f"**A good fit.** The curve follows the *pattern* and shrugs off "
        f"the noise; train ({train_mse:.3f}) and val ({val_mse:.3f}) are "
        "close, which is the signature of a model that generalises."
    )
else:
    st.error(
        f"**Overfitting territory.** Training MSE looks great "
        f"({train_mse:.3f}) but the validation MSE is worse "
        f"({val_mse:.3f}) — the wiggles between training points are the "
        "model chasing noise. Two live antidotes: (1) drop the degree; "
        "(2) raise the training points to 100 and watch the same degree "
        "behave better — more data pins the wiggles down. Antidote (3), "
        "regularization, is the next page."
    )

st.header("2 · The whole story in one chart: the U-curve")
st.markdown(
    "Fit **every** degree from 0 to 15 (done live just now) and plot both "
    "errors. This chart is one of the most important in all of ML — "
    "the shape appears for every model class, including deep survival "
    "networks, with 'flexibility' on the x-axis in some form."
)
degrees = np.arange(0, 16)
tr_curve, va_curve = [], []
for d in degrees:
    cf = fit_poly(x_train, y_train, d)
    tr_curve.append(np.mean((y_train - np.polyval(cf, x_train)) ** 2))
    va_curve.append(np.mean((y_val - np.polyval(cf, x_val)) ** 2))

best_d = int(degrees[np.argmin(va_curve)])
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.semilogy(degrees, tr_curve, marker="o", color="#1565c0",
            label="training MSE")
ax.semilogy(degrees, va_curve, marker="s", color="#e65100",
            label="validation MSE")
ax.axvline(best_d, color="#2e7d32", linestyle=":",
           label=f"best degree by validation = {best_d}")
ax.scatter([degree], [va_curve[degree]], s=140, facecolor="none",
           edgecolor="#2e7d32", linewidth=2, zorder=6,
           label=f"your slider (d={degree})")
ax.text(0.3, max(va_curve) * 0.5, "underfitting\n(both errors high)",
        fontsize=9, color="#546e7a")
ax.text(11.5, max(va_curve) * 0.5,
        "overfitting\n(train ↓, validation ↑)", fontsize=9, color="#546e7a")
ax.set_xlabel("polynomial degree (flexibility →)")
ax.set_ylabel("MSE (log scale)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
Read it left to right: **training error only ever goes down** as flexibility
grows (more wiggle can always hug the training points harder). Validation
error falls, bottoms out — degree {best_d} for this particular data — then
climbs as the model starts fitting noise. The validation set is the referee
that tells you when to stop; the training score alone would cheerfully march
you into the overfitting zone.

The statistician's framing: an inflexible model has **bias** (systematically
wrong — it can't represent the truth), an over-flexible one has **variance**
(re-draw the noise and you'd get a wildly different fit — it's at the mercy
of this sample's accidents). The sweet spot balances the two — the classic
**bias-variance trade-off**.
"""
)

st.header("3 · The same experiment in code")
show_example(
    '''import numpy as np, warnings

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)

def mse_of_degree(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # silence polyfit's grumbling
        coeffs = np.polyfit(x_train, y_train, d)
    tr = np.mean((y_train - np.polyval(coeffs, x_train)) ** 2)
    va = np.mean((y_val - np.polyval(coeffs, x_val)) ** 2)
    return tr, va

print(" d | train MSE | val MSE")
for d in [0, 1, 3, 5, 9, 15]:
    tr, va = mse_of_degree(d)
    print(f"{d:2d} | {tr:9.4f} | {va:8.4f}")''',
    """
- `make_data(n, seed)` — the true pattern (`np.sin(2πx)`) plus noise. Two different seeds give training and validation sets that share the *pattern* but have *independent noise* — exactly the situation a real model faces.
- `np.polyfit(x, y, d)` — least-squares-fits a degree-`d` polynomial (gradient descent's closed-form cousin from last page, generalised to curves). It returns the `d+1` coefficients.
- `np.polyval(coeffs, x)` — evaluates that polynomial at any `x`: the model's predictions.
- `warnings.catch_warnings()` — at high degrees polyfit warns the fit is numerically delicate; true, and part of the overfitting story, so we silence rather than hide from it.
- The printed table is the U-curve in numbers: watch the train column shrink monotonically while the val column falls then blows up.
""",
)

guided_sandbox(
    key="m5",
    steps="""
1. **Step 1** — write a function `val_mse(d)` that fits degree `d` on the
   training data (`np.polyfit`) and returns the MSE on the validation data
   (`np.polyval` + `np.mean`).
2. **Step 2** — loop `d` over `range(16)`, collecting `val_mse(d)` into a
   list, and print the best degree with `np.argmin`.
3. **Step 3** — print the *training* MSE for degree 15 next to its
   validation MSE. Say out loud what those two numbers mean. 😄
4. **Step 4 (stretch)** — regenerate the training data with `n=100` and
   re-run your loop. Does the best degree change? Does degree 15's
   validation MSE improve?
""",
    setup_code='''import numpy as np, warnings
warnings.simplefilter("ignore")   # silence high-degree polyfit warnings

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)
print("data ready:", len(x_train), "train points,", len(x_val), "val points")

# Step 1: define val_mse(d)


# Step 2: loop d in range(16), find the best degree


# Step 3: compare train vs val MSE at degree 15


# Step 4 (stretch): remake training data with n=100 and re-run
''',
    solution_code='''import numpy as np, warnings
warnings.simplefilter("ignore")

def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + r.normal(0, 0.3, n)

x_train, y_train = make_data(25, seed=5)
x_val, y_val = make_data(30, seed=99)

def val_mse(d):
    coeffs = np.polyfit(x_train, y_train, d)
    return np.mean((y_val - np.polyval(coeffs, x_val)) ** 2)

scores = [val_mse(d) for d in range(16)]
best = int(np.argmin(scores))
print(f"best degree: {best} (val MSE {scores[best]:.4f})")

c15 = np.polyfit(x_train, y_train, 15)
tr15 = np.mean((y_train - np.polyval(c15, x_train)) ** 2)
print(f"degree 15: train {tr15:.5f} vs val {scores[15]:.2f}  <- memorised!")

x_train, y_train = make_data(100, seed=5)
scores100 = [val_mse(d) for d in range(16)]
best100 = int(np.argmin(scores100))
print(f"with 100 training points: best degree {best100}, "
      f"degree-15 val MSE now {scores100[15]:.4f}")''',
)
