import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("😴 Lazy Evaluation & the Execution Plan")
st.markdown(
    """
This is the most important page in the section, and the idea that most confuses
people coming from pandas.

**Spark does not do what you tell it, when you tell it.**

When you write `df.filter(...)`, **nothing happens**. No data is read. No rows
are filtered. Spark simply writes down *"the user wants a filter here"* and hands
you back a new DataFrame that is really just a **recipe**. You can stack a dozen
such steps and Spark will still have touched precisely zero rows.

Only when you ask for an actual **result** — a count, a display, a save — does
Spark look at the whole recipe, optimise it, and run it.

- **Transformations** (lazy, free): `select`, `filter`, `withColumn`,
  `groupBy`, `join`, `orderBy`
- **Actions** (eager, expensive): `count`, `show`, `collect`, `toPandas`,
  `write`
"""
)

res = artifacts.load("s3_lazy")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

steps = pd.read_csv(io.StringIO(res["steps_csv"]))
t_transform = float(res["t_transform_total"])
t_action = float(res["t_action"])

st.header("1 · Watch the plan build up — then watch it fire")
st.markdown(
    "Drag the slider to add one transformation at a time. **Nothing runs.** "
    "The plan just grows. Then push it to the end and trigger the action."
)

n_steps = len(steps)
stage = st.slider("build the query…", 0, n_steps + 1, 0,
                  format="step %d")

labels = ["read parquet (1,351,138 rows)"] + steps["transformation"].tolist()
fired = stage > n_steps

fig, ax = plt.subplots(figsize=(10, 4.6))
for i, label in enumerate(labels):
    built = i <= stage
    y = len(labels) - 1 - i
    face = ("#c8e6c9" if fired else "#fff9c4") if built else "#eceff1"
    edge = ("#2e7d32" if fired else "#f9a825") if built else "#cfd8dc"
    ax.add_patch(Rectangle((1.6, y * 0.62), 5.6, 0.5, facecolor=face,
                           edgecolor=edge, linewidth=1.8))
    ax.text(1.85, y * 0.62 + 0.25, label, va="center", fontsize=9,
            color="#263238" if built else "#b0bec5")
    if built:
        ax.text(7.35, y * 0.62 + 0.25,
                "✅ EXECUTED" if fired else "📝 planned only — 0 rows touched",
                va="center", fontsize=7.5,
                color="#2e7d32" if fired else "#f57f17")
    if i < len(labels) - 1 and i < stage:
        ax.add_patch(FancyArrowPatch(
            (1.5, y * 0.62 + 0.25), (1.5, (y - 1) * 0.62 + 0.25),
            arrowstyle="-|>", mutation_scale=12,
            color="#2e7d32" if fired else "#f9a825"))

if fired:
    ax.add_patch(Rectangle((1.6, -0.85), 5.6, 0.55, facecolor="#c62828",
                           edgecolor="#8e0000", linewidth=2))
    ax.text(4.4, -0.58, "ACTION: .toPandas()  ⚡ NOW everything runs",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")
elif stage == n_steps:
    ax.add_patch(Rectangle((1.6, -0.85), 5.6, 0.55, facecolor="#eceff1",
                           edgecolor="#b0bec5", linewidth=1.6,
                           linestyle="--"))
    ax.text(4.4, -0.58, "…still nothing has run. Add the action →",
            ha="center", va="center", fontsize=9, color="#78909c")
ax.set_xlim(0, 10.2)
ax.set_ylim(-1.1, len(labels) * 0.62 + 0.1)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

if not fired:
    st.info(
        f"**{stage} step(s) planned. Zero rows read.** Spark has built a "
        "recipe and is waiting. You could add fifty more transformations and "
        "it would still be waiting."
    )
else:
    st.success(
        f"**The action fired the whole plan at once.** Spark read the parquet, "
        f"applied every filter, aggregated and sorted — in "
        f"**{t_action:.2f} s**."
    )

st.header("2 · The timings prove it")
fig, ax = plt.subplots(figsize=(9, 2.8))
ys = list(steps["transformation"]) + ["⚡ ACTION: .toPandas()"]
vals = list(steps["seconds"] * 1000) + [t_action * 1000]
colours = ["#ffd54f"] * len(steps) + ["#c62828"]
ax.barh(ys[::-1], vals[::-1], color=colours[::-1], edgecolor="white")
for i, v in enumerate(vals[::-1]):
    ax.text(v + 20, i, f"{v:.0f} ms", va="center", fontsize=8.5)
ax.set_xlabel("milliseconds")
ax.set_xlim(0, max(vals) * 1.22)
st.pyplot(fig)
plt.close(fig)

st.warning(
    f"""
**Read this honestly, because the naive story is slightly wrong.**

The five transformations cost **{t_transform * 1000:.0f} ms in total**, and the
single action cost **{t_action * 1000:.0f} ms** — about
**{t_action / max(t_transform, 1e-9):.1f}×** more, on its own.

But notice the transformations aren't *literally* free. Spark spends a few
milliseconds on each one **analysing** the plan: checking the column exists,
checking the types line up, resolving names. That's why you get an error
immediately if you `filter` on a column that doesn't exist — even though no data
has been read.

The correct statement is not *"transformations take no time"*. It is:

> **Transformations cost a fixed, tiny planning overhead that does not depend on
> the size of your data. Actions cost time proportional to your data.**

Put 100× more data in and the yellow bars barely move; the red bar grows 100×.
That is the distinction that matters.
"""
)

st.header("3 · Why bother? Because Spark rewrites your query")
st.markdown(
    """
Laziness isn't laziness for its own sake — it buys Spark the chance to see your
**whole** query before running any of it, and then to **improve it**. The
optimiser is called **Catalyst**, and here is its real output for our query:
"""
)
st.code(res["explain"], language="text")

st.markdown(
    """
Two optimisations to hunt for in that plan — they're the ones interviewers ask
about:

- **`PushedFilters: [...]`** — Spark took your `year >= 2015` filter and pushed
  it **down into the parquet reader**. The 2014 rows are never decoded at all;
  they're skipped at the file level. In pandas you'd have loaded all 1.35M rows
  into memory and *then* thrown most of them away.
- **`ReadSchema: struct<...>`** — look at how few columns it lists. Our table
  has 15 columns; the plan reads only the ones the query actually needs. This is
  **column pruning**, and it's free because parquet is columnar.

Neither of these could happen if Spark ran each command the moment you typed it.
**Laziness is what makes the optimisation possible.**
"""
)

with st.expander("See the full formatted plan (what you'd read when debugging)"):
    st.code(res["explain_formatted"], language="text")

st.header("4 · In code")
show_example(
    experiments.SNIPPETS["s3_lazy"],
    """
- Every line in the `plan = (...)` block is a **transformation**. Spark records each and returns a new DataFrame object immediately. The printed time confirms it.
- `.toPandas()` is the **action**. This is where Spark finally reads the file, applies the pushed-down filters, computes the aggregation and returns.
- ⚠️ **`.toPandas()` pulls the result into the driver's memory.** It's fine here — we asked for 5 rows. Calling it on a 100 GB DataFrame will kill your driver. Use `.limit()` first, or `.write.parquet(...)` to save the result distributed.
- **The debugging habit this creates:** if a Spark job seems to "hang", look at the last **action**, not the last line of code. That's where the work is actually happening.
""",
    key="s3_lazy",
    heavy=True,
    est="~30 s",
)

guided_sandbox(
    key="s3",
    heavy=True,
    defer=True,
    est="~40 s",
    steps="""
1. **Step 1** — build a chain of 4+ transformations on the deliveries table
   (filter, withColumn, groupBy, orderBy) and **time it**. Print how long the
   whole chain took to *build*. It should be milliseconds.
2. **Step 2** — now call an action (`.count()` or `.toPandas()`) and time
   *that*. Compare the two numbers.
3. **Step 3** — call `.explain()` on your plan **before** the action. Find the
   `PushedFilters` line. Which of your filters did Spark push into the file
   reader? Which columns is it actually reading (`ReadSchema`)?
4. **Step 4 (stretch)** — prove that analysis is eager even though execution is
   lazy: try to `filter` on a column that **doesn't exist**
   (e.g. `F.col("banana") > 1`). You get an error *immediately*, with no
   action called. Spark validated your plan without reading any data.
""",
    setup_code='''import time
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)
print("columns:", df.columns)

# Step 1: build a chain of transformations; time how long BUILDING takes


# Step 2: call an action; time that


# Step 3: .explain() the plan - find PushedFilters and ReadSchema


# Step 4 (stretch): filter on a column that doesn't exist -> instant error?


spark.stop()
''',
    solution_code='''import time
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

t0 = time.time()
plan = (df.filter(F.col("year") >= 2015)
          .filter(F.col("is_wide") == 0)
          .withColumn("is_boundary", (F.col("runs_off_bat") >= 4).cast("int"))
          .groupBy("batting_team")
          .agg(F.sum("is_boundary").alias("boundaries"))
          .orderBy(F.col("boundaries").desc()))
t_build = time.time() - t0
print(f"BUILDING the plan: {1000 * t_build:.0f} ms  (zero rows read)")

t0 = time.time()
result = plan.limit(5).toPandas()
t_action = time.time() - t0
print(f"the ACTION:        {t_action:.2f} s")
print(f"the action cost {t_action / max(t_build, 1e-9):.0f}x more\\n")
print(result.to_string(index=False))

print("\\n--- the optimised plan ---")
plan.explain()
print("look for PushedFilters (the year filter ran INSIDE the file reader)")
print("and ReadSchema (only the needed columns were read)")

print("\\n--- analysis is EAGER even though execution is lazy ---")
try:
    df.filter(F.col("banana") > 1)      # no action called!
except Exception as e:
    print("instant error, no data read:", str(e)[:90])

spark.stop()''',
)
