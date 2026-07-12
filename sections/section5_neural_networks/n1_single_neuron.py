import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.linear_model import LogisticRegression

from utils.mockdata import pumps, pumps_xor
from utils.sandbox import guided_sandbox, show_example

st.title("⚡ Logistic Regression as a Single Neuron")
st.markdown(
    """
Welcome to neural networks. The punchline of this whole section is that a
"neural network" is **not** a new kind of model — it's a *stack* of a model
you already know. This page shows that the humble logistic regression from
the Model Evaluation page **is a one-neuron neural network**, viewed in a
different mirror.

New scenario, new data (no more patients — we're in a machine shop now):
120 industrial **pumps** on a test rig, each measured on **vibration** and
**temperature** (both scaled 0–10), labelled `0` = passed inspection or
`1` = failed within a month. We want the probability a pump fails.

A **neuron** does exactly three things:

1. **Weigh the evidence**: multiply each input by a weight and add them,
   plus a bias: $z = w_1 \\cdot \\text{vib} + w_2 \\cdot \\text{temp} + b$.
   ($z$ gets called the *pre-activation*, or the neuron's "score".)
2. **Squash the score** through the **sigmoid** function
   $\\sigma(z) = 1/(1+e^{-z})$, which bends any number into (0, 1).
3. **Report the result** as a probability.

Weigh, squash, report. That's a neuron — and it's also *precisely* logistic
regression: same formula, same loss, same everything. "Training the neuron"
= finding $w_1, w_2, b$ by gradient descent on a loss (here: the log-loss
whose log-likelihood you met on the AIC page).
"""
)

X, y = pumps()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
clf = LogisticRegression().fit(X_tr, y_tr)
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]

st.header("1 · Numbers flowing through the neuron, live")
st.markdown(
    "The weights below are the *real* fitted values (trained just now on 80 "
    "pumps). Feed the neuron a pump with the sliders and follow its three "
    "steps left to right:"
)
c1, c2 = st.columns(2)
vib = c1.slider("vibration reading", 0.0, 10.0, 6.5, 0.1)
temp = c2.slider("temperature reading", 0.0, 10.0, 3.0, 0.1)

z = w1 * vib + w2 * temp + b
p = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(10, 3.6))
# input nodes
for (yy, label, val) in [(2.4, "vibration", vib), (0.8, "temperature", temp)]:
    ax.add_patch(Circle((1.0, yy), 0.42, facecolor="#e3f2fd",
                        edgecolor="#1565c0", linewidth=2))
    ax.text(1.0, yy, f"{val:.1f}", ha="center", va="center", fontsize=11,
            fontweight="bold")
    ax.text(1.0, yy - 0.62, label, ha="center", fontsize=9, color="#546e7a")
# sum node
ax.add_patch(Circle((4.4, 1.6), 0.55, facecolor="#fff3e0",
                    edgecolor="#e65100", linewidth=2))
ax.text(4.4, 1.6, f"z =\n{z:+.2f}", ha="center", va="center", fontsize=10)
ax.text(4.4, 0.78, "weigh the evidence", ha="center", fontsize=8,
        color="#546e7a")
# sigmoid node
ax.add_patch(Circle((7.2, 1.6), 0.55, facecolor="#ede7f6",
                    edgecolor="#4527a0", linewidth=2))
ax.text(7.2, 1.6, "σ", ha="center", va="center", fontsize=16)
ax.text(7.2, 0.78, "squash into (0,1)", ha="center", fontsize=8,
        color="#546e7a")
# output
ax.add_patch(Circle((9.3, 1.6), 0.55,
                    facecolor="#ffcdd2" if p > 0.5 else "#c8e6c9",
                    edgecolor="#37474f", linewidth=2))
ax.text(9.3, 1.6, f"{p:.2f}", ha="center", va="center", fontsize=11,
        fontweight="bold")
ax.text(9.3, 0.78, "P(fail)", ha="center", fontsize=8, color="#546e7a")
# edges with weights
for (y0, w, name) in [(2.4, w1, "w₁"), (0.8, w2, "w₂")]:
    ax.add_patch(FancyArrowPatch((1.45, y0), (3.9, 1.62), arrowstyle="-|>",
                                 mutation_scale=14, color="#78909c",
                                 connectionstyle="arc3,rad=0.08"))
    ax.text(2.6, (y0 + 1.6) / 2 + 0.18, f"× {name} = {w:+.2f}", fontsize=9,
            color="#37474f")
ax.add_patch(FancyArrowPatch((4.4, 3.1), (4.4, 2.2), arrowstyle="-|>",
                             mutation_scale=13, color="#78909c"))
ax.text(4.4, 3.3, f"+ bias b = {b:+.2f}", ha="center", fontsize=9,
        color="#37474f")
ax.add_patch(FancyArrowPatch((5.0, 1.6), (6.6, 1.6), arrowstyle="-|>",
                             mutation_scale=14, color="#78909c"))
ax.add_patch(FancyArrowPatch((7.8, 1.6), (8.7, 1.6), arrowstyle="-|>",
                             mutation_scale=14, color="#78909c"))
ax.set_xlim(0, 10.4)
ax.set_ylim(0.1, 3.7)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

sk_p = clf.predict_proba([[vib, temp]])[0, 1]
st.success(
    f"By hand: z = {w1:+.2f}×{vib:.1f} {w2:+.2f}×{temp:.1f} {b:+.2f} = "
    f"**{z:+.2f}**, then σ({z:+.2f}) = **{p:.3f}**. sklearn's "
    f"`predict_proba` for the same pump: **{sk_p:.3f}** — identical, "
    "because logistic regression *is* this neuron."
)

st.markdown(
    "The sigmoid is the bridge between the score and a probability — "
    "your current pump's position rides along it:"
)
zs = np.linspace(-8, 8, 300)
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(zs, 1 / (1 + np.exp(-zs)), color="#4527a0", linewidth=2.2)
ax.scatter([z], [p], s=120, color="#e65100", zorder=5,
           label=f"this pump: σ({z:.2f}) = {p:.2f}")
ax.axhline(0.5, color="#b0bec5", linestyle=":")
ax.axvline(0, color="#b0bec5", linestyle=":")
ax.set_xlabel("score z")
ax.set_ylabel("σ(z) = P(fail)")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    "Note the geometry: **z = 0 means σ = 0.5**, the fence-sitting point — "
    "so the decision boundary is the line where `z = 0`, which is a "
    "*straight line* in sensor space. And out in the tails (|z| > 4) the "
    "curve is nearly flat — the neuron is so sure that nothing moves it. "
    "That flatness will come back to haunt us on the backpropagation page."
)

st.header("2 · One neuron's world view — and the wall it hits")
st.markdown(
    "Left: our pump data, where one straight line does fine. Right: a nastier "
    "batch (`pumps_xor`) where a pump fails when **exactly one** reading is "
    "high — two faults that mask each other when combined. Both models below "
    "are real fits, computed live:"
)

Xx, yx = pumps_xor()
clf_xor = LogisticRegression().fit(Xx, yx)
gx, gy = np.meshgrid(np.linspace(0, 10, 160), np.linspace(0, 10, 160))
grid = np.column_stack([gx.ravel(), gy.ravel()])

c1, c2 = st.columns(2)
for col, (Xd, yd, model, title) in zip(
        [c1, c2],
        [(X_tr, y_tr, clf, "normal batch — a line suffices"),
         (Xx, yx, clf_xor, "XOR batch — the single neuron's wall")]):
    with col:
        zz = model.predict(grid).reshape(gx.shape)
        acc = model.score(Xd, yd)
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=["#c8e6c9", "#ffcdd2"], alpha=0.75)
        for cls, colour in [(0, "#2e7d32"), (1, "#c62828")]:
            m = yd == cls
            ax.scatter(Xd[m, 0], Xd[m, 1], c=colour, s=22, edgecolor="white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("vibration")
        ax.set_ylabel("temperature")
        ax.set_title(f"{title}\naccuracy {acc:.0%}", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.error(
    f"On the XOR batch the fitted single neuron manages "
    f"{clf_xor.score(Xx, yx):.0%} — barely above coin-flipping, and no "
    "amount of training can save it: its boundary is a straight line *by "
    "construction*, and no straight line separates opposite corners. This "
    "wall is historically famous (it stalled neural-network research for "
    "years in the 1970s). The escape — wiring neurons **into layers** so "
    "the network can bend its boundary — is where the rest of this section "
    "goes. You already glimpsed the other escape route: the SVM page's "
    "kernel trick manufactured a new dimension by hand. Hidden layers will "
    "learn to manufacture their own."
)

st.header("3 · In code")
show_example(
    '''import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.mockdata import pumps

X, y = pumps()
X_tr, y_tr = X[:80], y[:80]
clf = LogisticRegression().fit(X_tr, y_tr)

w1, w2 = clf.coef_[0]
b = clf.intercept_[0]
print(f"learned weights: w1={w1:.3f} (vibration), w2={w2:.3f} "
      f"(temperature), b={b:.3f}")

def neuron(vib, temp):
    z = w1 * vib + w2 * temp + b          # weigh
    return 1 / (1 + np.exp(-z))          # squash

for pump in [(6.5, 3.0), (2.0, 2.5), (7.5, 8.0)]:
    mine = neuron(*pump)
    theirs = clf.predict_proba([list(pump)])[0, 1]
    print(f"pump {pump}: my neuron {mine:.4f}  sklearn {theirs:.4f}")''',
    """
- `clf.coef_[0]` / `clf.intercept_[0]` — the two weights and the bias: the *entire* fitted model is three numbers. Both weights come out positive — higher vibration and higher temperature each push `z` up, toward failure — and their sizes say who pushes harder.
- `def neuron(vib, temp)` — the whole forward computation in two lines: weigh, squash. There is genuinely nothing else inside.
- `neuron(*pump)` — the `*` unpacks the tuple into the two arguments (the Functions page's star, still earning its keep).
- The loop confirms our two-liner matches sklearn to 4 decimals on three different pumps. When a library feels like magic, rebuild its smallest unit — this habit is why the forward-pass and backprop pages will hold no terrors.
""",
)

guided_sandbox(
    key="n1",
    steps="""
1. **Step 1** — write `sigmoid(z)` returning `1 / (1 + np.exp(-z))`, and
   `neuron(vib, temp)` computing `sigmoid(w1*vib + w2*temp + b)` with the
   fitted weights the setup code provides.
2. **Step 2** — check yourself against sklearn for the pump `(5.0, 5.0)`,
   then find the score `z` for that pump. How close to the fence is it?
3. **Step 3** — vectorize: compute `z_all = X_va @ clf.coef_[0] +
   clf.intercept_[0]`, apply your sigmoid, threshold at 0.5, and print the
   validation accuracy — it should equal `clf.score(X_va, y_va)`.
4. **Step 4 (stretch)** — fit `LogisticRegression()` on the XOR batch
   (`Xx, yx` provided) and print its accuracy. Then print its accuracy on
   just the pumps with vibration > 5 — notice it does no better there
   either; the failure is structural, not local.
""",
    setup_code='''import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.mockdata import pumps, pumps_xor

X, y = pumps()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
Xx, yx = pumps_xor()
clf = LogisticRegression().fit(X_tr, y_tr)
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]
print(f"fitted: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")

# Step 1: define sigmoid(z) and neuron(vib, temp)


# Step 2: compare with clf.predict_proba([[5.0, 5.0]]); print z too


# Step 3: vectorized predictions on X_va; accuracy vs clf.score


# Step 4 (stretch): the neuron vs the XOR batch
''',
    solution_code='''import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.mockdata import pumps, pumps_xor

X, y = pumps()
X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
Xx, yx = pumps_xor()
clf = LogisticRegression().fit(X_tr, y_tr)
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def neuron(vib, temp):
    return sigmoid(w1 * vib + w2 * temp + b)

z_mid = w1 * 5.0 + w2 * 5.0 + b
print(f"pump (5,5): my neuron {neuron(5.0, 5.0):.4f}, "
      f"sklearn {clf.predict_proba([[5.0, 5.0]])[0, 1]:.4f}, z={z_mid:.3f}")

z_all = X_va @ clf.coef_[0] + clf.intercept_[0]
pred = (sigmoid(z_all) >= 0.5).astype(int)
print(f"my accuracy: {np.mean(pred == y_va):.0%}  "
      f"clf.score: {clf.score(X_va, y_va):.0%}")

xor_clf = LogisticRegression().fit(Xx, yx)
print(f"XOR batch accuracy: {xor_clf.score(Xx, yx):.0%}")
high_vib = Xx[:, 0] > 5
print(f"...and on high-vibration pumps only: "
      f"{xor_clf.score(Xx[high_vib], yx[high_vib]):.0%} - structural, "
      "not fixable by more data")''',
)
