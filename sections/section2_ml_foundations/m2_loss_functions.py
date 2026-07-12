import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from utils.sandbox import guided_sandbox, show_example

st.title("📉 Loss Functions")
st.markdown(
    """
Last page you shrank the "average squared miss" by hand. That number has a
proper name: a **loss function** — a function that takes a model's
predictions and the true answers, and returns **one single number measuring
how bad the model is**. Lower = better; 0 = perfect.

Why one number? Because the machine can't "eyeball" a scatter plot like you
did. A search needs a score to minimise. The loss function *defines what
"good" means* — choose it badly and the machine optimises the wrong thing.
The two classic losses for regression:

| Loss | Formula (per point, then averaged) | Personality |
|---|---|---|
| **MSE** — mean squared error | $(y - \\hat{y})^2$ | punishes big misses *ferociously* (a miss of 10 costs 100) |
| **MAE** — mean absolute error | $\\|y - \\hat{y}\\|$ | every unit of miss costs the same |

($y$ = true answer, $\\hat{y}$ = "y-hat" = the model's prediction.)
"""
)

st.header("1 · See the loss as actual areas")
st.markdown(
    """
Eight mock data points (seed 3). Move the line. Each **red square's area is
literally one point's squared error** — MSE is the average of those areas.
Watch how one bad miss produces a square that dwarfs all the others.
"""
)

rng = np.random.default_rng(3)
x = np.linspace(0.5, 10, 8)
y = 0.9 * x + 2 + rng.normal(0, 1.1, 8)

c1, c2 = st.columns(2)
w = c1.slider("slope w", -1.0, 3.0, 0.4, 0.05)
b = c2.slider("intercept b", -2.0, 8.0, 5.0, 0.25)
pred = w * x + b
residuals = y - pred
mse = np.mean(residuals ** 2)
mae = np.mean(np.abs(residuals))

fig, ax = plt.subplots(figsize=(6.8, 6.2))
xs = np.linspace(-1, 12, 20)
ax.plot(xs, w * xs + b, color="#1565c0", linewidth=2.2,
        label=f"prediction line ŷ = {w:.2f}x + {b:.2f}")
for xi, yi, ri in zip(x, y, residuals):
    side = abs(ri)
    ax.add_patch(Rectangle((xi, min(yi, yi - ri)), side, side,
                           facecolor="#ef5350", alpha=0.30,
                           edgecolor="#c62828"))
    ax.plot([xi, xi], [yi, yi - ri], color="#c62828", linewidth=1.4)
ax.scatter(x, y, color="#263238", zorder=5, s=45, label="data points")
ax.set_aspect("equal")
ax.set_xlim(-1, 13)
ax.set_ylim(-2, 14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("squared error = the AREA of each square", fontsize=11)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("MSE (mean of the square areas)", f"{mse:.2f}")
c2.metric("MAE (mean of the red stick lengths)", f"{mae:.2f}")

st.header("2 · The loss *landscape*: loss as a function of the parameter")
st.markdown(
    """
Now the key mental flip for the whole course. Forget x and y for a second:
treat the **loss itself as a function of the slope `w`** (holding `b` at
your slider value). Every possible slope gets a loss — plotting that gives a
*landscape*, and "learning" = finding its lowest point. Your current slider
position is the orange dot.
"""
)

w_grid = np.linspace(-1, 3, 201)
preds_grid = w_grid[:, None] * x[None, :] + b
mse_curve = np.mean((y[None, :] - preds_grid) ** 2, axis=1)
mae_curve = np.mean(np.abs(y[None, :] - preds_grid), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for ax_, curve, cur, name in [(axes[0], mse_curve, mse, "MSE"),
                              (axes[1], mae_curve, mae, "MAE")]:
    ax_.plot(w_grid, curve, color="#1565c0", linewidth=2)
    ax_.scatter([w], [cur], color="#e65100", zorder=5, s=70,
                label=f"your w = {w:.2f}")
    best_w = w_grid[np.argmin(curve)]
    ax_.scatter([best_w], [curve.min()], color="#2e7d32", zorder=5, s=70,
                marker="*", label=f"lowest at w ≈ {best_w:.2f}")
    ax_.set_xlabel("slope w")
    ax_.set_ylabel("loss")
    ax_.set_title(f"{name} landscape (b fixed at {b:.2f})", fontsize=10)
    ax_.legend(fontsize=8)
    ax_.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
Two things to notice, both of which matter enormously later:

- **The MSE curve is a smooth bowl** — at every point it has a well-defined
  slope pointing "downhill". Gradient descent (next page) *walks down that
  slope*, which is why MSE-like losses are beloved.
- **The MAE curve is made of straight segments with kinks.** At a kink the
  "downhill direction" is ambiguous — optimisable, but less conveniently.
"""
)

st.header("3 · Why the choice matters: send in an outlier")
outlier_on = st.checkbox(
    "Add one outlier: a point at (9.5, 0.5) — e.g. a data-entry error")

x2 = np.append(x, 9.5) if outlier_on else x
y2 = np.append(y, 0.5) if outlier_on else y

# Best line under each loss, found honestly by brute-force grid search:
w_g = np.linspace(-1, 3, 161)
b_g = np.linspace(-2, 8, 161)
P = (w_g[:, None, None] * x2[None, None, :]
     + b_g[None, :, None])                       # every (w, b) line at once
R = y2[None, None, :] - P
mse_grid = (R ** 2).mean(axis=2)
mae_grid = np.abs(R).mean(axis=2)
iw, ib = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
w_mse, b_mse = w_g[iw], b_g[ib]
iw, ib = np.unravel_index(np.argmin(mae_grid), mae_grid.shape)
w_mae, b_mae = w_g[iw], b_g[ib]

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, color="#263238", s=45, zorder=5, label="data")
if outlier_on:
    ax.scatter([9.5], [0.5], color="#c62828", s=110, marker="X", zorder=6,
               label="the outlier")
xs = np.linspace(0, 11, 20)
ax.plot(xs, w_mse * xs + b_mse, color="#e65100", linewidth=2.2,
        label=f"best line by MSE: ŷ = {w_mse:.2f}x + {b_mse:.2f}")
ax.plot(xs, w_mae * xs + b_mae, color="#2e7d32", linewidth=2.2,
        linestyle="--", label=f"best line by MAE: ŷ = {w_mae:.2f}x + {b_mae:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
ax.set_title("each line is the true minimiser of its loss "
             "(found by a live grid search over 25,921 candidate lines)",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)
if outlier_on:
    st.info(
        f"The outlier **dragged the MSE line down** (slope fell to "
        f"{w_mse:.2f}) because its huge squared area dominates the average. "
        f"The MAE line (slope {w_mae:.2f}) barely moved — one point's "
        "absolute miss is just one vote among nine. Neither is 'right': if "
        "the outlier is a typo you want MAE's stubbornness; if it's a real "
        "patient you may want MSE's attention."
    )
else:
    st.info("Without the outlier the two lines nearly agree. Tick the box "
            "and watch them separate.")

st.header("4 · The same losses in code")
show_example(
    '''import numpy as np

y_true = np.array([10.0, 12.0, 15.0, 11.0])
y_pred = np.array([11.0, 12.5, 14.0, 21.0])   # last prediction is way off

errors = y_true - y_pred
print("errors:        ", errors)

mse = np.mean(errors ** 2)
mae = np.mean(np.abs(errors))
print("MSE:", round(mse, 2), "   MAE:", round(mae, 2))

# Where does each loss say the badness lives?
print("squared errors:", errors ** 2)
print("absolute errors:", np.abs(errors))''',
    """
- `errors = y_true - y_pred` — vectorized: all four misses at once. Sign says direction (negative = we over-predicted); both losses will erase the sign.
- `np.mean(errors ** 2)` — square, then average: MSE. The `**` and `mean` are the whole implementation.
- `np.mean(np.abs(errors))` — absolute value, then average: MAE.
- The last two prints show the mechanism of the outlier demo above: the `-10` miss contributes `100` to MSE's sum (drowning the others' 1, 0.25, 1) but only `10` to MAE's.
""",
)

guided_sandbox(
    key="m2",
    steps="""
1. **Step 1** — write `mse(y_true, y_pred)` and `mae(y_true, y_pred)`
   functions using `np.mean`, `**2` and `np.abs`. Print both for the data
   given in the editor.
2. **Step 2** — write `rmse(y_true, y_pred)`: the **square root** of the MSE
   (`np.sqrt`). Print it. Notice it's back in the *units of y* (marks, not
   marks²) — that's why papers often report RMSE.
3. **Step 3** — change the last value of `y_pred` from `21.0` to `11.5`
   (close to the truth, 11.0) and re-run. Which loss shrank by a bigger
   *factor*? Explain to yourself why.
4. **Step 4 (stretch)** — loop over candidate slopes `np.arange(0, 2, 0.1)`
   for the line `y_pred = slope * x_data`, and print the slope that
   minimises your `mse`. You just wrote a (crude) training algorithm.
""",
    setup_code='''import numpy as np

y_true = np.array([10.0, 12.0, 15.0, 11.0])
y_pred = np.array([11.0, 12.5, 14.0, 21.0])

# for step 4:
x_data = np.array([2.0, 4.0, 6.0, 8.0])
y_data = np.array([2.1, 3.8, 6.4, 7.9])

# Step 1: define mse(...) and mae(...), print both for y_true vs y_pred


# Step 2: define rmse(...) and print it


# Step 3: fix the bad prediction and re-run - watch both losses drop


# Step 4 (stretch): grid-search the slope minimising mse on x_data/y_data
''',
    solution_code='''import numpy as np

y_true = np.array([10.0, 12.0, 15.0, 11.0])
y_pred = np.array([11.0, 12.5, 14.0, 21.0])
x_data = np.array([2.0, 4.0, 6.0, 8.0])
y_data = np.array([2.1, 3.8, 6.4, 7.9])

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

print("MSE :", round(mse(y_true, y_pred), 3))
print("MAE :", round(mae(y_true, y_pred), 3))
print("RMSE:", round(rmse(y_true, y_pred), 3))

fixed = y_pred.copy()
fixed[-1] = 11.5
print("after fixing the bad prediction:")
print("MSE :", round(mse(y_true, fixed), 3),
      f"(shrank {mse(y_true, y_pred) / mse(y_true, fixed):.1f}x)")
print("MAE :", round(mae(y_true, fixed), 3),
      f"(shrank {mae(y_true, y_pred) / mae(y_true, fixed):.1f}x)")

best_s, best_loss = None, float("inf")
for s in np.arange(0, 2, 0.1):
    loss = mse(y_data, s * x_data)
    if loss < best_loss:
        best_s, best_loss = s, loss
print(f"best slope: {best_s:.1f} with MSE {best_loss:.3f}")''',
)
