import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

from utils.mockdata import patients_split
from utils.sandbox import guided_sandbox, show_example

st.title("🚀 Boosting: AdaBoost & Gradient Boosting")
st.markdown(
    """
Bagging trained its trees **in parallel** — independent, then averaged, to
cut *variance*. Boosting is the opposite temperament: train weak models
**one after another, each one focused on the mistakes of the team so far**,
to cut *bias*. A committee of specialists built sequentially, not a crowd of
generalists polled at once.

Two classic ways to say "focus on the mistakes":

- **AdaBoost (1997)** — give every training point a **weight**. After each
  round, *misclassified points get their weights increased*, so the next
  weak model is forced to care about exactly the cases the team keeps
  getting wrong.
- **Gradient boosting** — fit the next model to the **residuals** (what's
  left of the error) of the team so far, then add a small fraction of it to
  the team. Each round subtracts a slice of the remaining error.

The weak model in both is usually a tiny tree (often a one-split **stump**).
Boosting's magic is that a chain of models each barely better than guessing
can become extremely accurate — and, unlike bagging, **boosting can overfit
if you let the chain run too long**, which is why its knobs matter.
"""
)

X_train, X_val, y_train, y_val = patients_split()

st.header("1 · AdaBoost, hand-built: watch the weights swell")
st.markdown(
    "This AdaBoost is implemented from scratch on this page (stumps, "
    "weights, votes — no sklearn), so every number is inspectable. **Dot "
    "size = the weight each patient carries entering the chosen round.** "
    "The dashed line is that round's stump; the shading is the whole team's "
    "vote so far."
)

T = 10
y_pm = 2 * y_train - 1                       # labels as -1 / +1
thresholds = np.linspace(0.25, 9.75, 39)


def stump_predict(f, t, polarity, X):
    raw = np.where(X[:, f] > t, 1, -1)
    return polarity * raw


# --- the real AdaBoost loop, storing full history ---
w = np.ones(len(X_train)) / len(X_train)
history = []                                  # (weights_in, stump, alpha, err)
for round_i in range(T):
    best = None
    for f in [0, 1]:
        for t in thresholds:
            for pol in [1, -1]:
                err = w[stump_predict(f, t, pol, X_train) != y_pm].sum()
                if best is None or err < best[0]:
                    best = (err, f, t, pol)
    err, f, t, pol = best
    err = np.clip(err, 1e-10, 1 - 1e-10)
    alpha = 0.5 * np.log((1 - err) / err)
    history.append((w.copy(), (f, t, pol), alpha, err))
    w = w * np.exp(-alpha * y_pm * stump_predict(f, t, pol, X_train))
    w = w / w.sum()

round_shown = st.slider("round", 1, T, 1)
w_in, (f, t, pol), alpha, err = history[round_shown - 1]

gx, gy = np.meshgrid(np.linspace(0, 10, 150), np.linspace(0, 10, 150))
grid = np.column_stack([gx.ravel(), gy.ravel()])
team_grid = np.zeros(len(grid))
team_train = np.zeros(len(X_train))
team_val = np.zeros(len(X_val))
for (w_h, (fh, th, ph), a_h, _) in history[:round_shown]:
    team_grid += a_h * stump_predict(fh, th, ph, grid)
    team_train += a_h * stump_predict(fh, th, ph, X_train)
    team_val += a_h * stump_predict(fh, th, ph, X_val)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contourf(gx, gy, np.sign(team_grid).reshape(gx.shape),
            levels=[-1.5, 0, 1.5], colors=["#bbdefb", "#ffcdd2"], alpha=0.7)
for cls, colour in [(0, "#1565c0"), (1, "#c62828")]:
    m = y_train == cls
    ax.scatter(X_train[m, 0], X_train[m, 1], c=colour,
               s=3000 * w_in[m], edgecolor="white", zorder=3, alpha=0.85)
if f == 0:
    ax.axvline(t, color="#2e7d32", linestyle="--", linewidth=2.2)
else:
    ax.axhline(t, color="#2e7d32", linestyle="--", linewidth=2.2)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("biomarker A")
ax.set_ylabel("biomarker B")
feat_name = "A" if f == 0 else "B"
ax.set_title(f"round {round_shown}: stump asks “biomarker {feat_name} > "
             f"{t:.2f}?” — weighted error {err:.3f}, voice α = {alpha:.2f}",
             fontsize=10)
st.pyplot(fig)
plt.close(fig)

train_acc = np.mean(np.sign(team_train) == y_pm)
val_acc = np.mean(np.sign(team_val) == (2 * y_val - 1))
c1, c2, c3 = st.columns(3)
c1.metric("this stump's say in the vote (α)", f"{alpha:.2f}")
c2.metric(f"team of {round_shown}: train accuracy", f"{train_acc:.0%}")
c3.metric("validation accuracy", f"{val_acc:.0%}")
st.markdown(
    """
Walk the slider from 1 to 10 and watch the mechanism do its thing:

- **Round 1**: all dots equal (uniform weights); the stump is just the best
  single question — a one-question tree.
- **Rounds 2–3**: patients the previous stumps got wrong have *visibly
  swollen*; the new stump often cuts on the **other biomarker** to rescue
  them. Correct-and-easy patients shrivel to specks.
- The team's boundary (the shading) grows staircase corners a single stump
  could never make. Each stump's vote is scaled by its **α** — accurate
  stumps (low weighted error) speak louder.
- The weight-update rule is beautifully compact: multiply each point's
  weight by $e^{-\\alpha \\cdot y \\cdot \\hat y}$ (labels as ±1) — wrong
  predictions make the exponent positive (weight grows), right ones make it
  negative (weight shrinks). Then renormalise.
"""
)

st.header("2 · Gradient boosting: fit the residuals, add a slice, repeat")
st.markdown(
    """
For regression the "focus on mistakes" idea gets even more literal. Start
with a constant prediction (the mean). Round after round: compute the
**residuals** (truth − current prediction), fit a small tree *to the
residuals*, and add `learning_rate ×` that tree to the running prediction.
The wavy mock data from Section 2 returns (seed 5); every curve is a real
staged model from `GradientBoostingRegressor`:
"""
)

r5 = np.random.default_rng(5)
x_gb = np.sort(r5.uniform(0, 1, 40))
y_gb = np.sin(2 * np.pi * x_gb) + r5.normal(0, 0.3, 40)
lr_gb = st.select_slider("learning_rate (size of each added slice)",
                         [0.05, 0.1, 0.5, 1.0], value=0.1)
stage = st.select_slider("rounds so far", [1, 2, 5, 10, 25, 50, 100],
                         value=2)

gbr = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                learning_rate=lr_gb, random_state=0)
gbr.fit(x_gb.reshape(-1, 1), y_gb)
grid_x = np.linspace(0, 1, 300).reshape(-1, 1)
staged_grid = list(gbr.staged_predict(grid_x))
staged_train = list(gbr.staged_predict(x_gb.reshape(-1, 1)))
pred_grid = staged_grid[stage - 1]
pred_train = staged_train[stage - 1]
residuals = y_gb - pred_train

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].scatter(x_gb, y_gb, color="#37474f", s=26, zorder=5, label="data")
axes[0].plot(grid_x.ravel(), np.sin(2 * np.pi * grid_x.ravel()),
             color="#9e9e9e", linestyle="--", linewidth=1.4,
             label="true pattern")
axes[0].plot(grid_x.ravel(), pred_grid, color="#e65100", linewidth=2.2,
             label=f"team after {stage} round(s)")
axes[0].set_ylim(-1.9, 1.9)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_title("the running prediction", fontsize=10)
axes[1].stem(x_gb, residuals)
axes[1].set_ylim(-1.9, 1.9)
axes[1].grid(alpha=0.25)
axes[1].set_title(f"residuals after {stage} round(s) — the NEXT tree's "
                  "training target", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.markdown(
    f"""
At `learning_rate = {lr_gb}`, walk the rounds slider: the orange staircase
creeps toward the wave while the residual stems shrink toward zero — each
round literally eats a slice of what's left. Now set `learning_rate = 1.0`
and push to 100 rounds: the curve goes twitchy, chasing individual noisy
points (overfitting). Small learning rate + more rounds is the
slow-and-steady recipe that wins in practice — the same "step size" wisdom
as gradient descent, which is not a coincidence: the "gradient" in the name
is because fitting residuals *is* stepping down the MSE gradient in
function space. (Modern celebrity versions of this algorithm: **XGBoost**
and **LightGBM** — the usual winners on tabular-data competitions.)
"""
)

st.header("3 · In code, with the overfit-if-you-let-it warning visible")
show_example(
    '''import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from utils.mockdata import patients_split

X_train, X_val, y_train, y_val = patients_split()

gbc = GradientBoostingClassifier(n_estimators=300, max_depth=2,
                                 learning_rate=0.1, random_state=0)
gbc.fit(X_train, y_train)

# staged_predict = the team's prediction after every round:
val_acc = [np.mean(p == y_val) for p in gbc.staged_predict(X_val)]
train_acc = [np.mean(p == y_train) for p in gbc.staged_predict(X_train)]

for rounds in [1, 5, 25, 100, 300]:
    print(f"after {rounds:3d} rounds: train {train_acc[rounds-1]:.0%}  "
          f"val {val_acc[rounds-1]:.0%}")
best = int(np.argmax(val_acc)) + 1
print(f"validation peaked at round {best} ({max(val_acc):.0%})")''',
    """
- `GradientBoostingClassifier(...)` — same residual-chasing idea adapted for classification (it boosts on a classification loss's gradients rather than raw residuals). The three knobs shown are *the* three that matter: rounds, tree size, learning rate.
- `gbc.staged_predict(X_val)` — a generator yielding the team's predictions after round 1, 2, 3, …: the whole training history for the price of one fit.
- The printed table is the overfitting page's U-shape in a new costume: train accuracy marches to 100% while validation peaks and then decays. `n_estimators` **is a flexibility dial here** (contrast with the forest page, where more trees never hurt).
- `best = np.argmax(val_acc) + 1` — using validation to pick the stopping round = **early stopping**, boosting's standard defence, and a trick that returns for neural networks in Section 5.
""",
)

guided_sandbox(
    key="c6",
    steps="""
1. **Step 1** — fit `GradientBoostingRegressor(n_estimators=200,
   max_depth=2, learning_rate=0.1, random_state=0)` on the 1-D training
   data in the editor (`X_tr` is already reshaped for sklearn).
2. **Step 2** — using `staged_predict(X_va)`, build a list of validation
   MSEs (`np.mean((y_va - p) ** 2)` for each staged prediction `p`).
3. **Step 3** — print the best round (`np.argmin` + 1) and its MSE, plus
   the MSE at rounds 1 and 200 to see the full rise-fall-rise story.
4. **Step 4 (stretch)** — repeat with `learning_rate=1.0` and
   `learning_rate=0.05`. Which learning rate reaches the lowest validation
   MSE, and at how many rounds?
""",
    setup_code='''import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

r = np.random.default_rng(5)
x_tr = np.sort(r.uniform(0, 1, 40))
y_tr = np.sin(2 * np.pi * x_tr) + r.normal(0, 0.3, 40)
x_va = np.sort(r.uniform(0, 1, 30))
y_va = np.sin(2 * np.pi * x_va) + r.normal(0, 0.3, 30)
X_tr, X_va = x_tr.reshape(-1, 1), x_va.reshape(-1, 1)
print("data ready:", X_tr.shape, X_va.shape)

# Step 1: fit the GradientBoostingRegressor


# Step 2: validation MSE after every round via staged_predict


# Step 3: best round, plus MSE at rounds 1 and 200


# Step 4 (stretch): learning_rate 1.0 and 0.05 compared
''',
    solution_code='''import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

r = np.random.default_rng(5)
x_tr = np.sort(r.uniform(0, 1, 40))
y_tr = np.sin(2 * np.pi * x_tr) + r.normal(0, 0.3, 40)
x_va = np.sort(r.uniform(0, 1, 30))
y_va = np.sin(2 * np.pi * x_va) + r.normal(0, 0.3, 30)
X_tr, X_va = x_tr.reshape(-1, 1), x_va.reshape(-1, 1)

def staged_val_mse(lr):
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                    learning_rate=lr, random_state=0)
    gbr.fit(X_tr, y_tr)
    return [np.mean((y_va - p) ** 2) for p in gbr.staged_predict(X_va)]

mses = staged_val_mse(0.1)
best = int(np.argmin(mses))
print(f"lr=0.1: best at round {best + 1} (val MSE {mses[best]:.4f})")
print(f"        round 1: {mses[0]:.4f}   round 200: {mses[-1]:.4f}")

for lr in [1.0, 0.05]:
    m = staged_val_mse(lr)
    b = int(np.argmin(m))
    print(f"lr={lr:4}: best round {b + 1:3d}, val MSE {m[b]:.4f}, "
          f"final (200) {m[-1]:.4f}")''',
)
