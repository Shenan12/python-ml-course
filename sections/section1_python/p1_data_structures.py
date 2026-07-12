import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import sandbox, show_example

st.title("📦 Data Structures & Comprehensions")
st.markdown(
    """
So far you know **variables** — a name attached to *one* value. Real programs
juggle *many* values at once: 500 patients, 20 exam marks, a menu of prices.
Python gives you four built-in containers for this, and choosing the right
one is a genuine job-interview question:

| Container | Looks like | Ordered? | Changeable? | Duplicates? | Use it for |
|---|---|---|---|---|---|
| **list** | `[1, 2, 3]` | ✅ | ✅ | ✅ | an ordered sequence you'll add to / edit |
| **tuple** | `(1, 2, 3)` | ✅ | ❌ | ✅ | a fixed group, e.g. coordinates `(x, y)` |
| **dict** | `{"age": 21}` | ✅ (by insertion) | ✅ | keys ❌ | looking things up **by name**, not position |
| **set** | `{1, 2, 3}` | ❌ | ✅ | ❌ | membership tests & de-duplication |
"""
)

tab_list, tab_dict, tab_tuple, tab_set = st.tabs(
    ["Lists", "Dictionaries", "Tuples", "Sets"]
)

# ----------------------------------------------------------------- LISTS
with tab_list:
    st.subheader("A list is a row of numbered boxes")
    st.markdown(
        """
Every item sits in a **numbered position** called its *index*. Two rules trip
up every beginner:

1. **Counting starts at 0**, not 1. The first item is `fruits[0]`.
2. **Negative indices count from the end**: `fruits[-1]` is the last item.

Move the slider and watch which box the index picks out — the answer shown is
Python actually evaluating `fruits[i]`, not a drawing.
"""
    )

    fruits = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
    idx = st.slider("Pick an index `i`, then read off `fruits[i]`",
                    min_value=-len(fruits), max_value=len(fruits) - 1, value=0)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    for i, fruit in enumerate(fruits):
        selected = (i == idx) or (i - len(fruits) == idx)
        ax.add_patch(Rectangle((i, 0), 0.94, 1, facecolor="#ffd54f" if selected
                               else "#e3f2fd", edgecolor="#37474f"))
        ax.text(i + 0.47, 0.5, fruit, ha="center", va="center", fontsize=10)
        ax.text(i + 0.47, 1.18, str(i), ha="center", fontsize=10,
                color="#1565c0", fontweight="bold")
        ax.text(i + 0.47, -0.22, str(i - len(fruits)), ha="center", fontsize=10,
                color="#c62828")
    ax.text(-0.15, 1.18, "index →", ha="right", fontsize=9, color="#1565c0")
    ax.text(-0.15, -0.22, "negative →", ha="right", fontsize=9, color="#c62828")
    ax.set_xlim(-1.6, len(fruits) + 0.2)
    ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    # This result is Python genuinely indexing the list right now:
    st.success(f"`fruits[{idx}]` → `'{fruits[idx]}'`")

    st.markdown("**The everyday list operations:**")
    show_example(
        '''fruits = ["apple", "banana", "cherry"]

fruits.append("date")        # add to the END
fruits[0] = "apricot"        # replace position 0
first_two = fruits[0:2]      # "slice": positions 0 and 1 (2 is NOT included)

print(fruits)
print("length:", len(fruits))
print("slice:", first_two)
print("is banana in the list?", "banana" in fruits)''',
        """
- `fruits = ["apple", "banana", "cherry"]` — creates a list of 3 strings and names it `fruits`.
- `fruits.append("date")` — `.append(...)` is a *method*: an action the list knows how to do to itself. It adds one item at the end.
- `fruits[0] = "apricot"` — square brackets on the **left** of `=` mean "replace what's in box 0".
- `first_two = fruits[0:2]` — a **slice** `[start:stop]` copies boxes `start` up to *but not including* `stop`. So `0:2` gives boxes 0 and 1.
- `print(fruits)` — shows the whole list after the changes.
- `len(fruits)` — the built-in `len` counts the items.
- `"banana" in fruits` — the `in` keyword asks "is this item anywhere in the list?" and gives back `True` or `False`.
""",
    )

# ------------------------------------------------------------ DICTIONARIES
with tab_dict:
    st.subheader("A dictionary looks things up by *name*, not position")
    st.markdown(
        """
A **dict** stores `key: value` pairs. Instead of asking *"what's in box 3?"*
you ask *"what's the value for `'age'`?"*. Pick a key below — the arrow shows
the lookup, and the answer is Python really evaluating `student[key]`.
"""
    )

    student = {"name": "Priya", "degree": "Statistics", "year": 3,
               "average_mark": 68.5}
    chosen_key = st.selectbox("Look up student[...]", list(student.keys()))

    fig, ax = plt.subplots(figsize=(8, 2.8))
    for row, (k, v) in enumerate(student.items()):
        y = len(student) - 1 - row
        hit = k == chosen_key
        ax.add_patch(Rectangle((0, y), 2.6, 0.8,
                               facecolor="#ffd54f" if hit else "#e8f5e9",
                               edgecolor="#37474f"))
        ax.text(1.3, y + 0.4, repr(k), ha="center", va="center", fontsize=10)
        ax.add_patch(Rectangle((5.4, y), 2.6, 0.8,
                               facecolor="#ffd54f" if hit else "#fff3e0",
                               edgecolor="#37474f"))
        ax.text(6.7, y + 0.4, repr(v), ha="center", va="center", fontsize=10)
        ax.add_patch(FancyArrowPatch((2.65, y + 0.4), (5.35, y + 0.4),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color="#e65100" if hit else "#b0bec5",
                                     linewidth=2.2 if hit else 1))
    ax.text(1.3, len(student) + 0.25, "keys", ha="center", fontweight="bold")
    ax.text(6.7, len(student) + 0.25, "values", ha="center", fontweight="bold")
    ax.set_xlim(-0.3, 8.4)
    ax.set_ylim(-0.4, len(student) + 0.7)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.success(f"`student[{chosen_key!r}]` → `{student[chosen_key]!r}`")

    show_example(
        '''student = {"name": "Priya", "degree": "Statistics", "year": 3}

student["year"] = 4                  # change an existing value
student["favourite_module"] = "ML"   # brand-new key? it just gets added

print(student["name"])               # look up one value by its key
print(student.get("height", "not recorded"))  # safe lookup with a fallback
print(list(student.keys()))          # all the keys

for key, value in student.items():   # loop over pairs
    print(key, "→", value)''',
        """
- `student = {...}` — creates a dict. Each `key: value` pair is separated by commas.
- `student["year"] = 4` — same square-bracket-on-the-left trick as lists, but the "address" is a key, not a number.
- `student["favourite_module"] = "ML"` — assigning to a key that doesn't exist yet simply *creates* it.
- `student["name"]` — looks up the value stored under `"name"`. If the key didn't exist this would crash.
- `student.get("height", "not recorded")` — `.get` is the polite version: if the key is missing it returns your fallback instead of crashing.
- `student.keys()` — all keys. Wrapping in `list(...)` makes it print like a normal list.
- `for key, value in student.items():` — `.items()` hands you each pair; writing `key, value` unpacks the pair into two variables per loop turn.
""",
    )

# ----------------------------------------------------------------- TUPLES
with tab_tuple:
    st.subheader("A tuple is a list that refuses to change")
    st.markdown(
        """
Round brackets instead of square: `point = (3, 5)`. Reading works exactly like
a list (`point[0]`), but **writing is forbidden** — that's the whole point.
Use a tuple when the group of values *belongs together and should never be
edited by accident*: coordinates, an (r, g, b) colour, a database row.

The example below deliberately tries to break the rule, and shows the exact
error Python raises (caught so the page keeps running):
"""
    )
    show_example(
        '''point = (3, 5)
print("x is", point[0], "and y is", point[1])

x, y = point          # "unpacking": both variables filled in one line
print("unpacked:", x, y)

try:
    point[0] = 99     # forbidden!
except TypeError as err:
    print("Python said no:", err)''',
        """
- `point = (3, 5)` — round brackets make a tuple of two numbers.
- `point[0]` / `point[1]` — reading by index works exactly like a list.
- `x, y = point` — **unpacking**: Python matches the two names on the left to the two values in the tuple. You'll see this constantly in ML code (e.g. `X_train, X_test = ...`).
- `try:` / `except TypeError as err:` — we *attempt* the forbidden line; when Python raises a `TypeError`, the `except` block catches it and stores the message in `err`. (Error handling gets its own page soon — for now, just see that tuples genuinely can't be modified.)
""",
    )

# ------------------------------------------------------------------- SETS
with tab_set:
    st.subheader("A set is a bag of *unique* items — great for comparing groups")
    st.markdown(
        """
Curly brackets with just values: `{2, 3, 5}`. No order, no duplicates, and
lightning-fast *"is this in there?"* checks. Their superpower is comparing two
groups. Below, `A` and `B` are real Python sets — pick an operation and the
highlighted region *is* the result Python computed.
"""
    )
    A = {1, 2, 3, 4, 5, 6}
    B = {4, 5, 6, 7, 8}
    op = st.radio(
        "Operation",
        ["A | B  (union — in either)", "A & B  (intersection — in both)",
         "A - B  (difference — in A only)",
         "A ^ B  (symmetric difference — in exactly one)"],
        horizontal=True,
    )
    result = {"A |": A | B, "A &": A & B, "A -": A - B, "A ^": A ^ B}[op[:3]]

    only_a, both, only_b = A - B, A & B, B - A  # real set arithmetic
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for cx, colour in [(2.0, "#1565c0"), (3.6, "#2e7d32")]:
        circ = plt.Circle((cx, 1.7), 1.5, fill=False, linewidth=2,
                          edgecolor=colour)
        ax.add_patch(circ)
    ax.text(0.7, 3.25, "A", fontsize=14, color="#1565c0", fontweight="bold")
    ax.text(4.9, 3.25, "B", fontsize=14, color="#2e7d32", fontweight="bold")

    def place(items, x_center, spread=0.55):
        items = sorted(items)
        n = len(items)
        for j, val in enumerate(items):
            y = 1.7 + (j - (n - 1) / 2) * spread
            in_result = val in result  # highlight decided by the REAL result
            ax.text(x_center, y, str(val), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if in_result else "#78909c",
                    bbox=dict(boxstyle="circle,pad=0.28",
                              facecolor="#ef6c00" if in_result else "#eceff1",
                              edgecolor="none"))

    place(only_a, 1.35)
    place(both, 2.8)
    place(only_b, 4.25)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.1, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
    st.success(f"`{op[:5].strip()} B` → `{sorted(result)}` (orange circles)")

    show_example(
        '''marks = [70, 65, 70, 58, 65, 70]
unique_marks = set(marks)     # duplicates vanish
print(unique_marks)

module_a = {"Priya", "Tom", "Aisha"}
module_b = {"Tom", "Aisha", "Ben"}
print("taking both modules:", module_a & module_b)
print("only in module A:  ", module_a - module_b)''',
        """
- `set(marks)` — converting a list to a set throws away duplicates in one step. (Notice the printed order may differ from the list — sets don't keep order.)
- `module_a & module_b` — `&` is intersection: students in **both** sets.
- `module_a - module_b` — `-` is difference: in the first set but not the second.
""",
    )

st.divider()

# ---------------------------------------------------------- COMPREHENSIONS
st.header("List comprehensions: a loop, a filter and a transform in one line")
st.markdown(
    """
This pattern — *take a list, keep some items, transform them, collect the
results* — is so common that Python gives it a one-line syntax:

```python
squares = [n ** 2 for n in numbers if n % 2 == 0]
#          ^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
#          3. transform   1. loop      2. filter
```

Read it in the order the arrows suggest: **for** each `n` in `numbers`, **if**
`n` is even, put `n ** 2` into the new list. Drag the slider to watch the
machine process one item at a time — every keep/reject decision below is
Python actually evaluating `n % 2 == 0`, and every output value is a real
`n ** 2`.
"""
)

numbers = [3, 4, 7, 8, 10, 13, 14]
step = st.slider("Items processed so far", 0, len(numbers), len(numbers))

decisions = [(n, n % 2 == 0, n ** 2) for n in numbers]  # the real mechanism
output_so_far = [sq for (n, keep, sq) in decisions[:step] if keep]

fig, ax = plt.subplots(figsize=(9.5, 3.6))
for i, (n, keep, sq) in enumerate(decisions):
    processed = i < step
    face = ("#c8e6c9" if keep else "#ffcdd2") if processed else "#eceff1"
    ax.add_patch(Rectangle((i * 1.25, 2.6), 1.05, 0.8, facecolor=face,
                           edgecolor="#37474f"))
    ax.text(i * 1.25 + 0.52, 3.0, str(n), ha="center", va="center", fontsize=12)
    if processed:
        ax.text(i * 1.25 + 0.52, 2.35, "keep ✓" if keep else "reject ✗",
                ha="center", fontsize=8,
                color="#2e7d32" if keep else "#c62828")
ax.text(-0.15, 3.0, "input\n`numbers`", ha="right", va="center", fontsize=9)

ax.add_patch(Rectangle((2.4, 1.15), 3.9, 0.7, facecolor="#fff9c4",
                       edgecolor="#f9a825"))
ax.text(4.35, 1.5, "filter:  n % 2 == 0   →   transform:  n ** 2",
        ha="center", va="center", fontsize=10)

for j, sq in enumerate(output_so_far):
    ax.add_patch(Rectangle((j * 1.25, -0.4), 1.05, 0.8, facecolor="#bbdefb",
                           edgecolor="#37474f"))
    ax.text(j * 1.25 + 0.52, 0.0, str(sq), ha="center", va="center",
            fontsize=12, fontweight="bold")
ax.text(-0.15, 0.0, "output\n(so far)", ha="right", va="center", fontsize=9)

ax.set_xlim(-1.7, 9.2)
ax.set_ylim(-0.75, 3.75)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
st.success(f"After {step} of {len(numbers)} items: output = `{output_so_far}`")

show_example(
    '''numbers = [3, 4, 7, 8, 10, 13, 14]

# The long way: 4 lines
squares = []
for n in numbers:
    if n % 2 == 0:
        squares.append(n ** 2)
print("loop version:         ", squares)

# The comprehension: identical result, 1 line
squares2 = [n ** 2 for n in numbers if n % 2 == 0]
print("comprehension version:", squares2)

# They also work for dicts:
name_lengths = {name: len(name) for name in ["Priya", "Tom", "Aisha"]}
print(name_lengths)''',
    """
- Lines 3–6 — the classic pattern: start with an empty list, loop, test, append. Nothing new here.
- `squares2 = [n ** 2 for n in numbers if n % 2 == 0]` — the same three ingredients rearranged into one expression: the value to collect (`n ** 2`), the loop (`for n in numbers`), the filter (`if n % 2 == 0`). The `if` part is optional.
- `{name: len(name) for name in [...]}` — a **dict comprehension**: same idea, but you write a `key: value` pair before the `for`, and you get a dict back.
""",
)

sandbox(
    '''# The three-ingredient game: change any ingredient and re-run.
numbers = [3, 4, 7, 8, 10, 13, 14, 21, 22]

result = [n ** 2 for n in numbers if n % 2 == 0]
print("even numbers, squared:", result)

# Challenge 1: keep the ODD numbers instead (hint: != 0)
# Challenge 2: collect n + 100 instead of n ** 2
# Challenge 3 (harder): build a dict {n: n**2} for numbers bigger than 10

# A dict + list warm-up to edit too:
prices = {"coffee": 3.5, "tea": 2.8, "juice": 4.2}
cheap = [drink for drink, price in prices.items() if price < 4]
print("drinks under £4:", cheap)''',
    key="p1",
)
