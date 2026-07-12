from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import sandbox, show_example

st.title("📄 File I/O — reading and writing files")
st.markdown(
    """
Everything you've made so far vanished the moment the program ended, because
variables live in **memory**. Files live on **disk** and survive. Every ML
project starts by *reading* data from a file and ends by *writing* results to
one, so this page matters more than it looks.

**I/O** just means Input/Output — data coming into your program and going out.
"""
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
SAMPLE = DATA_DIR / "shopping_list.txt"
SAMPLE.write_text("milk\neggs\nbread\ntea\n", encoding="utf-8")

st.info(
    f"This page just created a real file on your disk at "
    f"`{SAMPLE}` containing four lines: milk, eggs, bread, tea. "
    "Everything below reads that actual file."
)

st.header("1 · What a text file really is: one long tape of characters")
st.markdown(
    """
On disk there are no "lines" — just one long sequence of characters, where a
special invisible character `\\n` ("newline") tells editors where to start a
new line. When you read a file, a **cursor** moves along the tape; each read
starts where the last one stopped.

Drag the slider: we genuinely `open()` the file and call `.read(n)`, then ask
the file object where its cursor is with `.tell()`.
"""
)

n_chars = st.slider("Read this many characters:  f.read(n)", 0, 20, 5)

# The real read:
with open(SAMPLE, encoding="utf-8") as f:
    grabbed = f.read(n_chars)
    cursor = f.tell()

content = SAMPLE.read_text(encoding="utf-8")
fig, ax = plt.subplots(figsize=(10, 1.7))
for i, ch in enumerate(content):
    is_nl = ch == "\n"
    consumed = i < cursor
    ax.add_patch(Rectangle((i * 0.5, 0.45), 0.46, 0.62,
                           facecolor=("#ffe0b2" if is_nl else "#c8e6c9")
                           if consumed else ("#fff8e1" if is_nl else "#eceff1"),
                           edgecolor="#78909c", linewidth=0.6))
    ax.text(i * 0.5 + 0.23, 0.76, "⏎" if is_nl else ch, ha="center",
            va="center", fontsize=10,
            color="#e65100" if is_nl else "#263238")
ax.add_patch(FancyArrowPatch((cursor * 0.5, 0.0), (cursor * 0.5, 0.42),
                             arrowstyle="-|>", mutation_scale=16,
                             color="#c62828", linewidth=2))
ax.text(cursor * 0.5, -0.22, f"cursor at position {cursor} (from f.tell())",
        ha="center", fontsize=9, color="#c62828")
ax.set_xlim(-0.3, len(content) * 0.5 + 0.3)
ax.set_ylim(-0.45, 1.3)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
st.success(f"`f.read({n_chars})` returned → `{grabbed!r}`")
st.markdown(
    "Notice the `\\n` characters (shown as ⏎) are *counted and consumed* like "
    "any other character — `'milk\\n'` is 5 characters, not 4."
)

st.header("2 · Writing, reading and appending — the `with open(...)` pattern")
show_example(
    f'''path = r"{SAMPLE}"

# WRITE mode "w": creates the file, or WIPES it if it already exists
with open(path, "w") as f:
    f.write("milk\\n")
    f.write("eggs\\n")

# APPEND mode "a": adds to the end, keeps what's there
with open(path, "a") as f:
    f.write("bread\\n")
    f.write("tea\\n")

# READ mode "r" (the default): three different ways to get the text out
with open(path) as f:
    whole = f.read()          # one big string
print("read() gave one string:", repr(whole))

with open(path) as f:
    lines = f.readlines()     # a LIST of lines (each keeps its \\n)
print("readlines() gave a list:", lines)

with open(path) as f:
    for line in f:            # the best way for big files: line by line
        print("item:", line.strip())''',
    """
- `with open(path, "w") as f:` — `open` gives you a *file object* `f`; the mode `"w"` means write. **The `with` block guarantees the file is properly closed when the block ends**, even if an error happens inside (that's `finally` from the last page, working for you behind the scenes). Always use `with`.
- `f.write("milk\\n")` — writes exactly what you give it. It does **not** add newlines for you, so we write the `\\n` ourselves.
- mode `"a"` — append: the cursor starts at the *end* of the existing tape.
- `f.read()` — with no number, reads the entire remaining tape into one string.
- `f.readlines()` — splits the tape at each `\\n` and returns a list of lines. Note each line still ends with its `\\n`.
- `for line in f:` — a file object is loopable! Each turn of the loop reads just one line, so even a 10 GB file never has to fit in memory at once.
- `line.strip()` — removes the trailing `\\n` (and any spaces) from each end of the string.
""",
)

st.header("3 · The file your dissertation will actually use: CSV")
st.markdown(
    """
Datasets nearly always arrive as **CSV** ("comma-separated values") — a text
file where the first line names the columns and each later line is one row.
You *could* read it with `open()` and split on commas… but pandas (two pages
from now) does it in one line. Here's a preview, executed for real:
"""
)
show_example(
    f'''import pandas as pd

csv_path = r"{DATA_DIR / 'patients.csv'}"

# Write a small CSV by hand, so you can see it is honestly just text:
with open(csv_path, "w") as f:
    f.write("name,age,blood_pressure\\n")
    f.write("Priya,34,118\\n")
    f.write("Tom,61,141\\n")
    f.write("Aisha,47,125\\n")

print(open(csv_path).read())        # the raw text on disk

df = pd.read_csv(csv_path)          # pandas parses it into a table
print(df)
print("average age:", df["age"].mean())''',
    """
- The `f.write` lines — build the CSV by hand: a header line, then one line per person, values separated by commas.
- `open(csv_path).read()` — proves the file is plain text. (Quick one-off reads like this are okay in a scratch script; in real code prefer the `with` form.)
- `pd.read_csv(csv_path)` — pandas reads the text, uses line 1 as column names, and gives you a **DataFrame** — a proper table that understands types (it worked out `age` is a number).
- `df["age"].mean()` — computes a column average in one step. This is the doorway into the Pandas page.
""",
)

sandbox(
    f'''# Your own file playground. This writes into the course's data folder.
path = r"{DATA_DIR / 'my_experiments.txt'}"

scores = [72, 85, 90, 66]

with open(path, "w") as f:
    f.write("experiment scores\\n")
    for s in scores:
        f.write(f"score: {{s}}\\n")

with open(path) as f:
    print(f.read())

# Challenge 1: append (mode "a") a final line saying "average: <the mean>"
#              - compute it with sum(scores) / len(scores).
# Challenge 2: read the file back and print ONLY lines containing a score
#              above 70 (hint: line.strip().split(": ")).
# Challenge 3: wrap a read of "does_not_exist.txt" in try/except
#              FileNotFoundError and print a friendly message instead.''',
    key="p5",
)
