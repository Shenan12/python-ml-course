import inspect

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import sandbox, show_example

st.title("🔧 Functions, args/kwargs & Lambdas")
st.markdown(
    """
You already know the basic `def`. This page upgrades that knowledge to what
real ML code demands, because a line like
`model.fit(X, y, epochs=10, verbose=False)` is nothing but function-argument
machinery — and by the end of this page you'll read it effortlessly.
"""
)

st.header("1 · Return values: the bit beginners mix up with print")
show_example(
    '''def add_vat(price, rate=0.20):
    """Return the price with VAT added."""
    return price * (1 + rate)

coffee = add_vat(3.50)          # rate not given -> the default 0.20 is used
fancy  = add_vat(3.50, 0.05)    # rate given -> 0.05 overrides the default

print("standard VAT:", coffee)
print("reduced VAT: ", fancy)''',
    """
- `def add_vat(price, rate=0.20):` — defines a function with two *parameters*. `rate=0.20` gives `rate` a **default value**, so callers may omit it.
- `\"\"\"Return the price...\"\"\"` — a *docstring*: a note explaining what the function does. Python ignores it; humans (and help tools) read it.
- `return price * (1 + rate)` — **`return` hands a value back to whoever called the function.** `print` only displays; `return` lets the caller *keep* the result in a variable. A function with no `return` hands back `None`.
- `coffee = add_vat(3.50)` — the call. `3.50` travels into `price`; `rate` silently becomes `0.20`.
- `fancy = add_vat(3.50, 0.05)` — both parameters supplied by position: first→`price`, second→`rate`.
""",
)

st.header("2 · Watch arguments travel into parameter slots")
st.markdown(
    """
When you call a function, Python plays a matching game: each argument you
supply must land in exactly one parameter slot. The function below uses every
kind of slot at once:

```python
def make_tea(kind, sugars=1, *extras, **notes):
```

- `kind` — required, filled by position or by name
- `sugars=1` — optional, has a default
- `*extras` — a **tuple** that scoops up any *spare positional* arguments
- `**notes` — a **dict** that scoops up any *spare keyword* arguments

Pick a call. The arrows show where each argument lands — and the mapping is
computed by Python's own `inspect` module doing the real binding, then the
function is genuinely called so you can see what it received.
"""
)


def make_tea(kind, sugars=1, *extras, **notes):
    return (f"kind={kind!r}, sugars={sugars!r}, "
            f"extras={extras!r}, notes={notes!r}")


CALLS = {
    'make_tea("earl grey")': (("earl grey",), {}),
    'make_tea("chai", 2)': (("chai", 2), {}),
    'make_tea("mint", 0, "honey", "lemon")': (("mint", 0, "honey", "lemon"), {}),
    'make_tea("green", temperature=80, cup="large")':
        (("green",), {"temperature": 80, "cup": "large"}),
    'make_tea("chai", 2, "honey", milk="oat")': (("chai", 2, "honey"),
                                                 {"milk": "oat"}),
}
call_text = st.selectbox("Choose a call", list(CALLS.keys()))
args, kwargs = CALLS[call_text]

# The REAL binding, done by Python itself:
bound = inspect.signature(make_tea).bind(*args, **kwargs)
bound.apply_defaults()

params = ["kind", "sugars", "*extras", "**notes"]
positional_named = ["kind", "sugars"]

# Where does each supplied argument land?
arrows = []          # (left_label, target_param, is_keyword)
for i, a in enumerate(args):
    target = positional_named[i] if i < len(positional_named) else "*extras"
    arrows.append((repr(a), target, False))
for k, v in kwargs.items():
    target = k if k in positional_named else "**notes"
    arrows.append((f"{k}={v!r}", target, True))

fig, ax = plt.subplots(figsize=(9, 3.4))
left_y = {j: 3.0 - j * 0.85 for j in range(len(arrows))}
right_y = {p: 3.0 - j * 0.85 for j, p in enumerate(params)}

for j, (label, target, is_kw) in enumerate(arrows):
    ax.add_patch(Rectangle((0, left_y[j] - 0.3), 2.9, 0.62,
                           facecolor="#e1f5fe" if not is_kw else "#f3e5f5",
                           edgecolor="#37474f"))
    ax.text(1.45, left_y[j], label, ha="center", va="center", fontsize=10)
    ax.text(2.95, left_y[j] + 0.02, " by name" if is_kw else " by position",
            fontsize=7, color="#78909c", va="center")
for p in params:
    ax.add_patch(Rectangle((6.1, right_y[p] - 0.3), 2.6, 0.62,
                           facecolor="#fff3e0", edgecolor="#37474f"))
    val = bound.arguments.get(p.strip("*"), "—")
    ax.text(7.4, right_y[p], f"{p} = {val!r}" if p not in ("*extras", "**notes")
            else f"{p} → {val!r}", ha="center", va="center", fontsize=9)
for j, (label, target, is_kw) in enumerate(arrows):
    ax.add_patch(FancyArrowPatch((3.9, left_y[j]), (6.05, right_y[target]),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#8e24aa" if is_kw else "#0277bd",
                                 linewidth=1.6,
                                 connectionstyle="arc3,rad=0.12"))
ax.text(1.45, 3.7, "arguments in the call", ha="center", fontweight="bold")
ax.text(7.4, 3.7, "parameter slots (after binding)", ha="center",
        fontweight="bold")
ax.set_xlim(-0.2, 9.0)
ax.set_ylim(3.0 - 3 * 0.85 - 0.6, 4.05)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.success(f"The function actually received → `{make_tea(*args, **kwargs)}`")
if "sugars" not in [a[1] for a in arrows]:
    st.info("`sugars` wasn't supplied, so its **default `1`** filled the slot.")

st.header("3 · Writing your own `*args` and `**kwargs`")
show_example(
    '''def describe_run(model_name, *metrics, **settings):
    print("model:", model_name)
    print("metrics received (a tuple):", metrics)
    print("settings received (a dict):", settings)
    for name, value in settings.items():
        print("  setting", name, "=", value)

describe_run("random_forest", 0.91, 0.88, depth=5, seed=42)''',
    """
- `def describe_run(model_name, *metrics, **settings):` — one required parameter, then `*metrics` collects any extra positional values into a **tuple**, and `**settings` collects any extra `name=value` pairs into a **dict**.
- In the call, `"random_forest"` fills `model_name`; the spare positionals `0.91, 0.88` become the tuple `metrics`; the spare keywords `depth=5, seed=42` become the dict `settings`.
- `for name, value in settings.items():` — exactly the dict-looping pattern from the previous page. Everything stacks.
- **Why this matters:** it's how libraries let you pass any option you like — when you later write `plt.plot(x, y, color="red", linewidth=2)`, those keywords are being scooped up by machinery just like this.
""",
)

st.header("4 · Lambdas: tiny nameless functions used as *arguments*")
st.markdown(
    """
A **lambda** is a one-line function with no name: `lambda x: x * 2` means
*"given `x`, produce `x * 2`"*. Its natural habitat is as the `key=` argument
of `sorted(...)`, which asks: *"before comparing items, what value should I
judge each item by?"*

Below, each student is a tuple `(name, mark, age)`. Choose a sort key — the
bars physically reorder because `sorted` is genuinely re-run with that lambda.
"""
)

students = [("Priya", 68, 22), ("Tom", 91, 24), ("Aisha", 75, 21),
            ("Ben", 55, 23), ("Zara", 82, 25)]
key_choice = st.radio(
    "Sort the students by…",
    ["mark (lambda s: s[1])", "name (lambda s: s[0])", "age (lambda s: s[2])"],
    horizontal=True,
)
key_fn = {"mark": lambda s: s[1], "name": lambda s: s[0],
          "age": lambda s: s[2]}[key_choice.split(" ")[0]]
ordered = sorted(students, key=key_fn)  # the real sort, using the real lambda

fig, ax = plt.subplots(figsize=(8, 2.8))
names = [s[0] for s in ordered]
marks = [s[1] for s in ordered]
bars = ax.bar(range(len(ordered)), marks, color="#42a5f5", edgecolor="#0d47a1")
for i, s in enumerate(ordered):
    ax.text(i, s[1] + 1.5, f"{s[0]}\nmark {s[1]}, age {s[2]}", ha="center",
            fontsize=8)
ax.set_xticks(range(len(ordered)))
ax.set_xticklabels(names)
ax.set_ylim(0, 108)
ax.set_ylabel("mark")
ax.set_title(f"sorted(students, key={key_choice.split('(')[1][:-1]})")
st.pyplot(fig)
plt.close(fig)

show_example(
    '''students = [("Priya", 68), ("Tom", 91), ("Aisha", 75)]

by_mark = sorted(students, key=lambda s: s[1])
print("lowest mark first:", by_mark)

top_first = sorted(students, key=lambda s: s[1], reverse=True)
print("highest mark first:", top_first)

double = lambda x: x * 2          # legal, but unusual — see note below
print("double(7) =", double(7))''',
    """
- `key=lambda s: s[1]` — for every student tuple `s`, `sorted` computes `s[1]` (the mark) and orders by *that*, while keeping whole tuples in the output.
- `reverse=True` — flips to descending order.
- `double = lambda x: x * 2` — you *can* name a lambda, but if you're naming it, standard style says just write `def double(x): return x * 2`. Lambdas shine when they're small and passed straight into another function.
""",
)

sandbox(
    '''# A mini grade-book built from this page's ideas.
students = [("Priya", 68, 22), ("Tom", 91, 24), ("Aisha", 75, 21),
            ("Ben", 55, 23), ("Zara", 82, 25)]

def classify(mark, first_cutoff=70, pass_cutoff=40):
    if mark >= first_cutoff:
        return "First"
    elif mark >= pass_cutoff:
        return "Pass"
    return "Fail"

for name, mark, age in sorted(students, key=lambda s: s[1], reverse=True):
    print(f"{name:6} mark={mark:3}  ->  {classify(mark)}")

# Challenge 1: sort by age instead of mark.
# Challenge 2: call classify with first_cutoff=80 - who loses their First?
# Challenge 3: write summarise(*marks) that prints the min, max and mean
#              of any number of marks, then call it with 5 numbers.''',
    key="p2",
)
