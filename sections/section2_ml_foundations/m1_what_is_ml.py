import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import guided_sandbox, show_example

st.title("🤖 What Machine Learning Is (vs Traditional Programming)")
st.markdown(
    """
In every program you have written so far, **you** supplied the rules:
*"if mark ≥ 70, print First."* Machine learning flips the arrow. You supply
**examples of inputs together with their correct answers**, and the computer
searches for the rule that connects them. The rule it finds is called a
**model**.

That's the entire idea. Everything else in this course — loss functions,
gradient descent, neural networks, DeepSurv — is just machinery for making
that *search for a rule* work well.
"""
)

fig, ax = plt.subplots(figsize=(9.5, 3.0))
for (y0, title, left, right, colour) in [
    (1.9, "TRADITIONAL PROGRAMMING", "rules (your code)\n+  data",
     "answers", "#1565c0"),
    (0.3, "MACHINE LEARNING", "data\n+  answers (examples)",
     "rules (the model)", "#e65100"),
]:
    ax.text(0.1, y0 + 1.08, title, fontsize=10, fontweight="bold",
            color=colour)
    ax.add_patch(Rectangle((0.1, y0), 3.4, 0.95, facecolor="#eceff1",
                           edgecolor=colour, linewidth=2))
    ax.text(1.8, y0 + 0.48, left, ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((6.1, y0), 3.4, 0.95, facecolor="#eceff1",
                           edgecolor=colour, linewidth=2))
    ax.text(7.8, y0 + 0.48, right, ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((3.6, y0 + 0.48), (5.95, y0 + 0.48),
                                 arrowstyle="-|>", mutation_scale=22,
                                 color=colour, linewidth=2.5))
    ax.text(4.78, y0 + 0.68, "the computer", ha="center", fontsize=8,
            color=colour)
ax.set_xlim(0, 9.8)
ax.set_ylim(0, 3.3)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.header("1 · Be the machine: find the rule yourself")
st.markdown(
    """
Below are 25 **mock students** (generated with a hidden true rule plus random
noise — seed 42, so it's the same every time). Each dot is one student:
hours studied per week vs final mark.

Your job is the *learning* part: move the two sliders to choose a line
`mark = slope × hours + intercept` that fits the cloud of dots as well as you
can. The **average squared miss** (how far your line is from the dots,
squared, averaged — formally introduced next page) is recomputed live as you
drag.
"""
)

rng = np.random.default_rng(42)
hours = rng.uniform(0, 12, 25)
marks = np.clip(38 + 3.6 * hours + rng.normal(0, 6, 25), 0, 100)

c1, c2 = st.columns(2)
slope = c1.slider("slope (extra marks per hour studied)", 0.0, 8.0, 1.0, 0.1)
intercept = c2.slider("intercept (mark for someone who studies 0 hours)",
                      0.0, 80.0, 60.0, 1.0)

your_pred = slope * hours + intercept
your_mse = np.mean((marks - your_pred) ** 2)

# The best possible line, found by NumPy (the "machine" answer):
X = np.column_stack([hours, np.ones_like(hours)])
(best_slope, best_intercept), *_ = np.linalg.lstsq(X, marks, rcond=None)
best_mse = np.mean((marks - (best_slope * hours + best_intercept)) ** 2)

show_machine = st.checkbox("Reveal the line the machine finds (in green)")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(hours, marks, color="#37474f", zorder=3, label="students (data)")
xs = np.linspace(0, 12, 50)
ax.plot(xs, slope * xs + intercept, color="#e65100", linewidth=2.5,
        label=f"YOUR rule: mark = {slope:.1f}·hours + {intercept:.0f}")
for h, m, p in zip(hours, marks, your_pred):
    ax.plot([h, h], [m, p], color="#ef9a9a", linewidth=1, zorder=1)
if show_machine:
    ax.plot(xs, best_slope * xs + best_intercept, color="#2e7d32",
            linewidth=2.5, linestyle="--",
            label=f"machine: mark = {best_slope:.2f}·hours + "
                  f"{best_intercept:.1f}")
ax.set_xlabel("hours studied per week")
ax.set_ylabel("final mark")
ax.set_ylim(0, 105)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("your average squared miss", f"{your_mse:.1f}")
c2.metric("the machine's (best possible for a line)",
          f"{best_mse:.1f}" if show_machine else "revealed by the checkbox")
st.markdown(
    """
The pink verticals are your line's *misses*. What you just did by hand —
nudging numbers to shrink the misses — **is literally what "learning" means
in machine learning**: adjusting a rule's numbers to fit examples. The two
numbers you tuned are called **parameters**, and the machine's search for
them (which found the green line instantly) is what we build, from scratch,
over the next two pages.
"""
)

st.header("2 · The same idea in code")
show_example(
    '''import numpy as np

# 25 example students: inputs (hours) and correct answers (marks)
rng = np.random.default_rng(42)
hours = rng.uniform(0, 12, 25)
marks = np.clip(38 + 3.6 * hours + rng.normal(0, 6, 25), 0, 100)

# A "model" with two parameters we can tune:
def predict(hours, slope, intercept):
    return slope * hours + intercept

# A score for how bad any given rule is:
def average_squared_miss(slope, intercept):
    misses = marks - predict(hours, slope, intercept)
    return np.mean(misses ** 2)

print("wild guess   (slope 1.0, intercept 60):",
      round(average_squared_miss(1.0, 60), 1))
print("better guess (slope 3.0, intercept 45):",
      round(average_squared_miss(3.0, 45), 1))
print("machine's answer via least squares:")
X = np.column_stack([hours, np.ones_like(hours)])
(best_slope, best_intercept), *_ = np.linalg.lstsq(X, marks, rcond=None)
print("  slope:", round(best_slope, 2), " intercept:",
      round(best_intercept, 1),
      " miss:", round(average_squared_miss(best_slope, best_intercept), 1))''',
    """
- `rng = np.random.default_rng(42)` — a random-number generator with a fixed **seed** (42), so the "random" data is identical every run. Reproducibility is a habit you'll keep for your dissertation.
- `hours = rng.uniform(0, 12, 25)` — 25 random values between 0 and 12: our mock inputs.
- `marks = np.clip(38 + 3.6*hours + rng.normal(0, 6, 25), 0, 100)` — the **hidden true rule** (38 plus 3.6 per hour) plus normally-distributed noise, clipped into 0–100. Real data always = pattern + noise.
- `def predict(...)` — the model: just a function with tunable numbers in it.
- `misses = marks - predict(...)` — vectorized subtraction (NumPy page!): every student's error at once.
- `np.mean(misses ** 2)` — square each miss, average them: one number that says how bad this rule is. Next page this gets its proper name: a **loss function**.
- `np.linalg.lstsq(X, marks, rcond=None)` — NumPy's built-in "find the best line" solver (least squares). We build `X` with a column of ones so the intercept gets learned too — that trick is explained properly on the gradient descent page.
- The printed numbers show the machine's answer beats both guesses.
""",
)

st.header("3 · Where this goes next")
st.markdown(
    """
| Question you should be asking | The page that answers it |
|---|---|
| *"Average squared miss" — why squared? What else could score badness?* | **Loss Functions** (next) |
| *How does the machine actually search for the best parameters?* | **Gradient Descent** |
| *How do we check the rule works on students it hasn't seen?* | **Train/Validation/Test Splits** |
| *What if the rule fits the examples too perfectly?* | **Overfitting vs Underfitting** |

One vocabulary note before moving on: this whole setup — learn from
input+answer examples — is called **supervised learning**, and it's the kind
your dissertation uses (survival models learn from patients whose outcomes
are known). Predicting a *number* (a mark) is **regression**; predicting a
*category* (pass/fail) is **classification**. There is also unsupervised
learning (finding structure with no answers given — Section 3's clustering
pages), but supervised is home base.
"""
)

guided_sandbox(
    key="m1",
    steps="""
1. **Step 1** — write a function `predict(hours, slope, intercept)` that
   returns `slope * hours + intercept`.
2. **Step 2** — write a function `score(slope, intercept)` that computes the
   misses (`marks - predict(...)`), squares them, and returns their mean
   (`np.mean`).
3. **Step 3** — call `score` with three different slope/intercept pairs of
   your choosing and `print` each result. Can you get below 40 by hand?
4. **Step 4 (stretch)** — loop over `slope` values `np.arange(0, 8, 0.5)`
   (keep `intercept = 38`) and print the slope with the smallest score.
""",
    setup_code='''import numpy as np

# The same 25 mock students as the page above (seed 42):
rng = np.random.default_rng(42)
hours = rng.uniform(0, 12, 25)
marks = np.clip(38 + 3.6 * hours + rng.normal(0, 6, 25), 0, 100)
print("data ready:", len(hours), "students")

# Step 1: define predict(hours, slope, intercept)


# Step 2: define score(slope, intercept)


# Step 3: try three slope/intercept pairs and print their scores


# Step 4 (stretch): loop slopes 0 to 7.5 in steps of 0.5, intercept=38,
#                   and print which slope scores lowest
''',
    solution_code='''import numpy as np

rng = np.random.default_rng(42)
hours = rng.uniform(0, 12, 25)
marks = np.clip(38 + 3.6 * hours + rng.normal(0, 6, 25), 0, 100)

def predict(hours, slope, intercept):
    return slope * hours + intercept

def score(slope, intercept):
    misses = marks - predict(hours, slope, intercept)
    return np.mean(misses ** 2)

print("try 1 (1.0, 60):", round(score(1.0, 60), 1))
print("try 2 (3.0, 45):", round(score(3.0, 45), 1))
print("try 3 (3.6, 38):", round(score(3.6, 38), 1))

best_slope, best_score = None, float("inf")
for s in np.arange(0, 8, 0.5):
    sc = score(s, 38)
    if sc < best_score:
        best_slope, best_score = s, sc
print(f"best slope with intercept 38: {best_slope} "
      f"(score {best_score:.1f})")''',
)
