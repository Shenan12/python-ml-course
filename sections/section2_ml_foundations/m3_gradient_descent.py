import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.sandbox import guided_sandbox, show_example

st.title("⛰️ Gradient Descent — walking down the loss landscape")
st.markdown(
    """
Last page you saw the loss **landscape**: every choice of parameters gets a
loss, and learning = finding the lowest point. Grid-searching every
combination worked for 2 parameters, but DeepSurv has *thousands* — a grid is
hopeless. **Gradient descent** is the search that scales, and it's the engine
inside every neural network in this course.

The idea fits in one sentence: **stand on the landscape, feel which way is
downhill, take a small step that way, repeat.**

- "Which way is downhill" is the **gradient** — calculus gives, for each
  parameter, the slope of the loss at your current position. The gradient
  points *uphill*, so we step in the opposite direction.
- "A small step" is the **learning rate** (written α or `lr`) — how far you
  move each time. You are about to see why this one number can make or break
  training.
- The update rule, for each parameter: `w = w - lr * gradient_of_w`.

Mock data for this whole page (seed 11): 40 students, `x` = hours studied
*relative to the class average*, `y` = mark, true rule `y = 2.5x + 12` plus
noise.
"""
)

rng = np.random.default_rng(11)
x = rng.uniform(-5, 5, 40)
y = 12 + 2.5 * x + rng.normal(0, 2.5, 40)


def loss_of(w, b):
    return np.mean((y - (w * x + b)) ** 2)


def run_gd(w0, b0, lr, n_steps):
    w, b = w0, b0
    path = [(w, b, loss_of(w, b))]
    for _ in range(n_steps):
        pred = w * x + b
        grad_w = -2 * np.mean(x * (y - pred))
        grad_b = -2 * np.mean(y - pred)
        w, b = w - lr * grad_w, b - lr * grad_b
        if not (np.isfinite(w) and np.isfinite(b)):
            path.append((w, b, np.inf))
            break
        path.append((w, b, loss_of(w, b)))
    return path


st.header("1 · One parameter first: the ball rolling into the bowl")
st.markdown(
    "Fix the intercept at `b = 12` and search only the slope `w`. The bowl "
    "is the MSE landscape from the last page; the dots are actual gradient "
    "descent steps, computed live. Try each learning rate — especially the "
    "biggest one."
)
c1, c2 = st.columns(2)
lr1 = c1.select_slider("learning rate (1-D walk)",
                       [0.002, 0.01, 0.04, 0.115, 0.15], value=0.01)
n1 = c2.slider("number of steps", 1, 40, 12)

w_walk, b_fixed = -1.5, 12.0
walk = [w_walk]
for _ in range(n1):
    grad = -2 * np.mean(x * (y - (walk[-1] * x + b_fixed)))
    w_next = walk[-1] - lr1 * grad
    walk.append(w_next)
    if not np.isfinite(w_next) or abs(w_next) > 1e6:
        break

w_grid = np.linspace(-2.5, 7.5, 300)
bowl = [loss_of(wi, b_fixed) for wi in w_grid]
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(w_grid, bowl, color="#1565c0", linewidth=2)
walk_arr = np.clip(np.array(walk), -2.5, 7.5)
walk_loss = [loss_of(wi, b_fixed) for wi in walk_arr]
ax.plot(walk_arr, walk_loss, color="#e65100", linewidth=1,
        marker="o", markersize=5, zorder=5)
for i in [0, 1, 2]:
    if i < len(walk_arr) - 1:
        ax.annotate(f"step {i}", (walk_arr[i], walk_loss[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8,
                    color="#e65100")
ax.set_xlabel("slope w")
ax.set_ylabel("MSE loss (b fixed at 12)")
diverged = (not np.isfinite(walk[-1])
            or loss_of(walk[-1], b_fixed) > loss_of(walk[0], b_fixed))
ax.set_title(f"lr = {lr1}: DIVERGING — each step lands HIGHER than the last"
             if diverged else
             f"lr = {lr1}: {len(walk) - 1} steps, "
             f"finishing at w = {walk[-1]:.3f}")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
if diverged:
    st.error(
        "**Divergence.** Each step overshot the bottom and landed *higher* "
        "on the other side, so the next gradient was even bigger — the "
        "steps snowballed. This is the classic too-big learning rate "
        "failure; every practitioner has met it."
    )
elif lr1 <= 0.002:
    st.info("Converging, but crawling — tiny steps waste compute. "
            "Learning-rate choice is a *trade-off*.")
else:
    st.success(f"Settled near the bottom of the bowl (w ≈ {walk[-1]:.2f}). "
               "Notice the steps automatically shrink near the bottom — "
               "the gradient itself gets smaller there, so `lr × gradient` "
               "does too.")

st.header("2 · Both parameters: the walk on the 2-D landscape")
st.markdown(
    "Now search slope **and** intercept together. The coloured rings are "
    "the real loss surface (darker = lower); the orange trail is gradient "
    "descent, starting from the ✕."
)
c1, c2 = st.columns(2)
lr2 = c1.select_slider("learning rate (2-D walk)",
                       [0.002, 0.01, 0.04, 0.115, 0.15], value=0.04)
n2 = c2.slider("number of steps ", 1, 300, 60)

path = run_gd(w0=-1.5, b0=2.0, lr=lr2, n_steps=n2)
pw = np.array([p[0] for p in path])
pb = np.array([p[1] for p in path])
finite = np.isfinite(pw) & np.isfinite(pb) & (np.abs(pw) < 50) & (np.abs(pb) < 200)

W, B = np.meshgrid(np.linspace(-3, 8, 90), np.linspace(-2, 26, 90))
Z = ((y[None, None, :] - (W[..., None] * x[None, None, :] + B[..., None]))
     ** 2).mean(axis=2)

# The exact answer, for comparison (derived properly in part 3):
X_design = np.column_stack([x, np.ones_like(x)])
(w_star, b_star), *_ = np.linalg.lstsq(X_design, y, rcond=None)

fig, ax = plt.subplots(figsize=(8.5, 5))
cs = ax.contourf(W, B, Z, levels=30, cmap="viridis_r", alpha=0.85)
fig.colorbar(cs, ax=ax, label="MSE loss")
ax.plot(pw[finite], pb[finite], color="#ff6d00", linewidth=1.5, marker="o",
        markersize=3.5, zorder=5, label="gradient descent path")
ax.scatter([pw[0]], [pb[0]], marker="X", s=130, color="white", zorder=6,
           edgecolor="black", label="start (-1.5, 2)")
ax.scatter([w_star], [b_star], marker="*", s=220, color="#ffd600", zorder=6,
           edgecolor="black", label=f"exact minimum ({w_star:.2f}, {b_star:.2f})")
ax.set_xlabel("slope w")
ax.set_ylabel("intercept b")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("the loss surface over (w, b), with the real descent path")
st.pyplot(fig)
plt.close(fig)

final_ok = np.isfinite(path[-1][2])
c1, c2, c3 = st.columns(3)
c1.metric("loss at start", f"{path[0][2]:.1f}")
c2.metric(f"loss after {len(path) - 1} steps",
          f"{path[-1][2]:.3f}" if final_ok else "∞ (diverged)")
c3.metric("loss at the exact minimum", f"{loss_of(w_star, b_star):.3f}")
st.markdown(
    """
Notice the path's *shape* at moderate learning rates: it drops fast along
the `w` direction, then crawls along the shallow `b` valley. The surface is
a stretched bowl because `w` and `b` affect the loss at different scales.
**Remember this picture** — it is exactly why the DeepSurv paper
standardizes its inputs before training: standardizing makes the bowl
rounder, and round bowls are easy to descend.
"""
)

st.header("3 · Checking ourselves twice: the maths and the code")
st.markdown(
    r"""
For MSE with a line, calculus gives the downhill directions in closed form
(differentiate $\frac{1}{n}\sum (y_i - (w x_i + b))^2$ with respect to each
parameter):

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum x_i\,(y_i - \hat y_i) \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum (y_i - \hat y_i)$$

You don't need to reproduce the differentiation — but you *should* distrust
formulas handed to you. The snippet below verifies the gradient two
independent ways, then verifies the whole algorithm against NumPy's exact
least-squares answer:
"""
)
show_example(
    '''import numpy as np

rng = np.random.default_rng(11)
x = rng.uniform(-5, 5, 40)
y = 12 + 2.5 * x + rng.normal(0, 2.5, 40)

def loss(w, b):
    return np.mean((y - (w * x + b)) ** 2)

# --- Check 1: the calculus formula vs a "nudge test" (finite difference).
# The slope at w SHOULD be roughly (loss(w+tiny) - loss(w-tiny)) / (2*tiny).
w0, b0, tiny = 1.0, 3.0, 1e-6
formula   = -2 * np.mean(x * (y - (w0 * x + b0)))
nudge     = (loss(w0 + tiny, b0) - loss(w0 - tiny, b0)) / (2 * tiny)
print("gradient by calculus formula:", round(formula, 6))
print("gradient by nudge test:      ", round(nudge, 6))

# --- Check 2: run full gradient descent, compare to the exact solution.
w, b, lr = 0.0, 0.0, 0.04
for step in range(400):
    pred = w * x + b
    w -= lr * (-2 * np.mean(x * (y - pred)))
    b -= lr * (-2 * np.mean(y - pred))
print(f"gradient descent found:  w={w:.4f}  b={b:.4f}  loss={loss(w, b):.4f}")

X = np.column_stack([x, np.ones_like(x)])
(w_ex, b_ex), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"exact (closed form):     w={w_ex:.4f}  b={b_ex:.4f}  "
      f"loss={loss(w_ex, b_ex):.4f}")''',
    """
- `def loss(w, b)` — the MSE from the previous page, as a reusable function.
- **The nudge test** — a gradient is just "how much does the loss change if I nudge this parameter a tiny bit?". So compute `loss(w+0.000001)` minus `loss(w-0.000001)`, divide by the nudge width, and you get the slope *without any calculus*. The two printed numbers agreeing (to ~6 decimals) is real evidence the formula is right — this exact trick, called a *finite-difference check*, is how professionals debug hand-written gradients.
- `w -= lr * (-2 * np.mean(x * (y - pred)))` — the update rule: current value, minus learning rate times gradient. `-=` is shorthand for `w = w - ...`.
- `np.linalg.lstsq(...)` — for a *linear* model, a formula (the "normal equations") jumps straight to the exact minimum, no walking needed. The design matrix `X` pairs each `x` with a constant `1`, so the same machinery learns `b` too (the 1's coefficient *is* the intercept).
- The final two lines agreeing is the honest punchline: **gradient descent works**. So why ever walk when you can jump? Because the jump formula *only exists for linear models*. DeepSurv's loss has no closed form — walking is the only option. That's why this page exists.
""",
)

guided_sandbox(
    key="m3",
    steps="""
1. **Step 1** — complete the `grad_w(w, b)` and `grad_b(w, b)` functions
   using the formulas above (each is one `return` line with `np.mean`).
2. **Step 2** — write the descent loop: start `w, b = 0.0, 0.0`, and for 200
   steps update both parameters with `lr = 0.04`. (Careful: compute both
   gradients *before* updating either parameter.)
3. **Step 3** — print the final `w`, `b` and `loss(w, b)`. Compare with the
   exact answer printed by the setup code.
4. **Step 4 (stretch)** — re-run with `lr = 0.15`. Print the loss every 20
   steps and watch the explosion happen live.
""",
    setup_code='''import numpy as np

rng = np.random.default_rng(11)
x = rng.uniform(-5, 5, 40)
y = 12 + 2.5 * x + rng.normal(0, 2.5, 40)

def loss(w, b):
    return np.mean((y - (w * x + b)) ** 2)

X = np.column_stack([x, np.ones_like(x)])
(w_ex, b_ex), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"target (exact): w={w_ex:.4f}  b={b_ex:.4f}")

# Step 1: complete these
def grad_w(w, b):
    ...  # return -2 * mean of x * (y - prediction)

def grad_b(w, b):
    ...

# Step 2: the loop - 200 steps, lr = 0.04


# Step 3: print your final w, b, loss


# Step 4 (stretch): retry with lr = 0.15, printing loss every 20 steps
''',
    solution_code='''import numpy as np

rng = np.random.default_rng(11)
x = rng.uniform(-5, 5, 40)
y = 12 + 2.5 * x + rng.normal(0, 2.5, 40)

def loss(w, b):
    return np.mean((y - (w * x + b)) ** 2)

X = np.column_stack([x, np.ones_like(x)])
(w_ex, b_ex), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"target (exact): w={w_ex:.4f}  b={b_ex:.4f}")

def grad_w(w, b):
    return -2 * np.mean(x * (y - (w * x + b)))

def grad_b(w, b):
    return -2 * np.mean(y - (w * x + b))

w, b = 0.0, 0.0
lr = 0.04
for step in range(200):
    gw, gb = grad_w(w, b), grad_b(w, b)   # both BEFORE updating either
    w, b = w - lr * gw, b - lr * gb
print(f"found by descent: w={w:.4f}  b={b:.4f}  loss={loss(w, b):.4f}")

print("\\nnow with lr = 0.15 (too big):")
w, b = 0.0, 0.0
for step in range(100):
    gw, gb = grad_w(w, b), grad_b(w, b)
    w, b = w - 0.15 * gw, b - 0.15 * gb
    if step % 20 == 0:
        print(f"  step {step:3d}: loss = {loss(w, b):.2e}")
    if not np.isfinite(loss(w, b)):
        print("  ...overflowed past what a float can hold. Divergence.")
        break''',
)
