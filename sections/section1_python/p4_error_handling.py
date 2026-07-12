import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import sandbox, show_example

st.title("🚨 Error Handling")
st.markdown(
    """
Errors are not failures — they're Python's way of *telling you precisely what
went wrong*. Professional code doesn't avoid errors; it **expects** them and
decides what should happen when they occur. That's what `try` / `except` is
for. In ML work you'll hit these constantly: a data file that doesn't exist,
a column that isn't numeric, a matrix whose shape doesn't fit.
"""
)

st.header("1 · Meet the errors you'll actually see")
st.markdown(
    """
Every row in this table was produced by **really running** the code in the
first column just now and catching what Python raised — the messages are
Python's own words, not paraphrases.
"""
)

demos = [
    ("print(speling)", "You used a name that doesn't exist (typo?)"),
    ('"3" + 4', "You mixed types that don't combine"),
    ("[1, 2, 3][10]", "You asked a list for a position it doesn't have"),
    ('{"a": 1}["b"]', "You asked a dict for a key it doesn't have"),
    ("10 / 0", "You divided by zero"),
    ('int("hello")', "The value can't be converted to the type you asked for"),
    ("open('no_such_file.txt')", "The file you tried to open isn't there"),
]
rows = []
for code, meaning in demos:
    try:
        eval(code)  # noqa: S307 - deliberately raising errors for teaching
        raised, msg = "(no error!)", ""
    except Exception as exc:  # noqa: BLE001
        raised, msg = type(exc).__name__, str(exc)
    rows.append({"code that ran": code, "error raised": raised,
                 "Python's message": msg, "in plain English": meaning})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.header("2 · try / except / else / finally — watch the path light up")
st.markdown(
    """
The full safety net has four blocks. Here is the exact code we're about to
run, where `text` is whatever you choose below:

```python
try:
    number = int(text)        # step 1: convert the text to a whole number
    result = 10 / number      # step 2: divide
except ValueError:
    print("that's not a whole number")
except ZeroDivisionError:
    print("can't divide by zero")
else:
    print("success:", result) # runs ONLY if the try block had no error
finally:
    print("this always runs") # runs no matter what — cleanup lives here
```

Choose an input. The code genuinely runs with your choice, and the diagram
highlights the blocks that *actually executed*:
"""
)

text = st.radio('Set `text` to…', ['"5"', '"0"', '"hello"', '"2.5"'],
                horizontal=True).strip('"')

ran = []            # every block appends its name when it truly executes
messages = []
try:
    ran.append("try (started)")
    number = int(text)
    result = 10 / number
    ran.append("try (finished cleanly)")
except ValueError as exc:
    ran.append("except ValueError")
    messages.append(f"caught ValueError: {exc}")
except ZeroDivisionError as exc:
    ran.append("except ZeroDivisionError")
    messages.append(f"caught ZeroDivisionError: {exc}")
else:
    ran.append("else")
    messages.append(f"success: result = {result}")
finally:
    ran.append("finally")

blocks = ["try (started)", "try (finished cleanly)", "except ValueError",
          "except ZeroDivisionError", "else", "finally"]
labels = {
    "try (started)": "try:  int(text)  then  10 / number",
    "try (finished cleanly)": "…try block reached its end without an error",
    "except ValueError": 'except ValueError:  "not a whole number"',
    "except ZeroDivisionError": 'except ZeroDivisionError:  "can\'t divide by 0"',
    "else": "else:  print the successful result",
    "finally": "finally:  always runs",
}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
y = len(blocks) - 1
positions = {}
for b in blocks:
    executed = b in ran
    ax.add_patch(Rectangle((0.4, y), 7.4, 0.72,
                           facecolor="#c8e6c9" if executed else "#eceff1",
                           edgecolor="#2e7d32" if executed else "#b0bec5",
                           linewidth=2 if executed else 1))
    ax.text(0.65, y + 0.36, labels[b], va="center", fontsize=10,
            color="#1b5e20" if executed else "#90a4ae")
    ax.text(8.0, y + 0.36, "✓ ran" if executed else "skipped", va="center",
            fontsize=9, color="#2e7d32" if executed else "#b0bec5")
    positions[b] = y
    y -= 1
order = [b for b in blocks if b in ran]
for a, b in zip(order, order[1:]):
    ax.add_patch(FancyArrowPatch((0.25, positions[a] + 0.36),
                                 (0.25, positions[b] + 0.36),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="#2e7d32", linewidth=1.8))
ax.set_xlim(0, 9.3)
ax.set_ylim(-0.4, len(blocks) + 0.2)
ax.axis("off")
ax.set_title(f'What actually happened when text = "{text}"', fontsize=11)
st.pyplot(fig)
plt.close(fig)
for m in messages:
    st.info(m)
st.markdown(
    """
Things to notice as you click through all four inputs:

- **`"5"`** — no error, so both `except` blocks are skipped and `else` runs.
- **`"0"`** — `int("0")` is fine, but `10 / 0` explodes; Python jumps
  straight to the *matching* `except` and never returns to finish `try`.
- **`"hello"`** and **`"2.5"`** — `int(...)` fails at step 1, so the division
  never even runs. (Yes, `int("2.5")` fails — Python won't silently round
  text for you.)
- **`finally` ran every single time.** That's its job: closing files,
  disconnecting from databases — cleanup that must happen no matter what.
"""
)

st.header("3 · Raising your own errors")
st.markdown(
    """
`try/except` *reacts* to problems; `raise` lets you *create* one on purpose
when someone misuses your code. Failing loudly and early beats silently
producing nonsense — an ML pipeline that accepts a negative age will happily
train a meaningless model without ever complaining.
"""
)
show_example(
    '''class InvalidMarkError(Exception):
    """Raised when a mark is outside 0-100."""
    pass

def letter_grade(mark):
    if not 0 <= mark <= 100:
        raise InvalidMarkError(f"mark must be 0-100, got {mark}")
    if mark >= 70:
        return "A"
    elif mark >= 40:
        return "Pass"
    return "Fail"

print(letter_grade(85))

try:
    print(letter_grade(140))
except InvalidMarkError as err:
    print("rejected:", err)''',
    """
- `class InvalidMarkError(Exception):` — a custom error is just a class that inherits from `Exception` (inheritance from the OOP page, already paying off). The body is only a docstring, so `pass` says "nothing more to add".
- `if not 0 <= mark <= 100:` — Python lets you chain comparisons; this reads exactly like maths notation.
- `raise InvalidMarkError(f"...")` — `raise` throws the error immediately; the function stops right there. The f-string puts the offending value in the message, which future-you will be grateful for.
- `letter_grade(85)` — a valid call, works normally.
- The `try/except` around `letter_grade(140)` — catches our own error type by name and prints its message instead of crashing.
""",
)

sandbox(
    '''# A robust "safe divide" - run it, then break it in interesting ways.
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "cannot divide by zero"
    except TypeError:
        return f"need two numbers, got {a!r} and {b!r}"

print(safe_divide(10, 4))
print(safe_divide(10, 0))
print(safe_divide(10, "carrot"))

# Challenge 1: add a call safe_divide("ten", 2) - which except catches it?
# Challenge 2: extend safe_divide so dividing by 0 returns float("inf")
#              instead of a message.
# Challenge 3: write check_age(age) that raises ValueError for ages
#              below 0 or above 120, and prove both directions work
#              using try/except.''',
    key="p4",
)
