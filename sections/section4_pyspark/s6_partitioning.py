import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🧩 Partitioning & Shuffles")
st.markdown(
    """
A **partition** is a chunk of your data that lives on one machine and is processed
by one task. Partitions are the atom of parallelism in Spark: **8 partitions = up
to 8 things happening at once.**

Almost every "why is my Spark job slow?" question has its answer on this page,
which is why it's worth your time even though the section is optional.
"""
)

res = artifacts.load("s6_partitions")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

default_parts = int(res["default_partitions"])
counts = pd.read_csv(io.StringIO(res["partition_counts_csv"]))

st.header("1 · A nasty surprise: our data has ONE partition")
c1, c2 = st.columns([1, 2])
c1.metric("partitions Spark chose", default_parts)
c2.markdown(
    f"""
Our parquet file is a **single small file**, so Spark read it into
**{default_parts} partition**. All {int(counts['count'].sum()):,} rows sit in one
chunk, processed by **one core**.

Every other core on your machine is **idle**. We have a "cluster" and we're using
one worker.
"""
)

fig, ax = plt.subplots(figsize=(10, 2.4))
for i in range(8):
    used = i < default_parts
    ax.add_patch(Rectangle((i * 1.24, 0.5), 1.05, 1.0,
                           facecolor="#c8e6c9" if used else "#eceff1",
                           edgecolor="#2e7d32" if used else "#cfd8dc",
                           linewidth=1.8))
    ax.text(i * 1.24 + 0.52, 1.15, f"core {i + 1}", ha="center", fontsize=8)
    ax.text(i * 1.24 + 0.52, 0.78,
            f"{int(counts['count'].sum()):,}\nrows" if used else "idle 😴",
            ha="center", fontsize=7.5,
            color="#1b5e20" if used else "#b0bec5")
ax.set_xlim(-0.2, 10.2)
ax.set_ylim(0.2, 1.8)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.warning(
    """
**This is a real, common production problem, and it has a name: the
small-files-vs-one-big-file trap.**

Spark decides partitions from **how the data is stored**. One file → one
partition (for a small file). A thousand tiny files → a thousand partitions, each
with a task's worth of overhead and almost no work. Neither is what you want.

**Rules of thumb worth remembering:**
- aim for partitions of roughly **100–200 MB** each,
- aim for **2–4 partitions per CPU core** so cores always have work queued,
- if your data arrives badly partitioned, **fix it explicitly** with
  `.repartition(n)`.
"""
)

st.header("2 · So just repartition and go 8× faster, right? Wrong.")
st.markdown(
    "Here's the same aggregation run at different partition counts — measured, "
    "not theorised:"
)
timings = pd.read_csv(io.StringIO(res["timings_csv"]))
fig, ax = plt.subplots(figsize=(8.5, 3.0))
ax.plot(timings["partitions"], timings["seconds"], marker="o", linewidth=2.2,
        color="#1565c0")
best = timings.loc[timings["seconds"].idxmin()]
ax.scatter([best["partitions"]], [best["seconds"]], s=200, marker="*",
           color="#2e7d32", zorder=5,
           label=f"fastest: {int(best['partitions'])} partitions")
ax.set_xlabel("number of partitions")
ax.set_ylabel("seconds")
ax.set_xscale("log", base=2)
ax.set_xticks(timings["partitions"])
ax.set_xticklabels(timings["partitions"])
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.dataframe(timings.round(3), hide_index=True, width="stretch")

slowest = timings.loc[timings["seconds"].idxmax()]
st.error(
    f"""
**Repartitioning barely helped — and too many partitions made it *worse*.**

Fastest: **{int(best['partitions'])} partitions** at {best['seconds']:.2f}s.
Slowest: **{int(slowest['partitions'])} partitions** at {slowest['seconds']:.2f}s.

Why? Two reasons, both of which generalise:

1. **`repartition()` is itself a shuffle.** To create 8 balanced partitions,
   Spark must physically move every row across the network (or, here, between
   threads). On 1.35M rows that costs more than the parallelism wins back.
2. **Every partition carries overhead** — a task to schedule, serialise,
   launch, and collect. At 16 partitions on this small dataset we're paying
   16 lots of overhead to split up work that takes a second anyway.

**The honest lesson: partitioning is not a free speed dial.** More partitions
help only when there is enough *work per partition* to justify the overhead. On
a 100 GB dataset across a real cluster the picture inverts completely — there,
under-partitioning is catastrophic. Know which regime you're in.
"""
)

st.header("3 · Narrow vs wide: the distinction that explains everything")
st.markdown(
    r"""
Every Spark operation is one of two kinds, and this is *the* concept to take
away from this page:
"""
)

fig, ax = plt.subplots(figsize=(10.5, 3.4))
ax.text(2.6, 3.1, "NARROW  (no shuffle)", ha="center", fontsize=10,
        fontweight="bold", color="#2e7d32")
for i in range(4):
    x = 0.7 + i * 1.0
    ax.add_patch(Rectangle((x, 1.9), 0.75, 0.65, facecolor="#c8e6c9",
                           edgecolor="#2e7d32"))
    ax.add_patch(Rectangle((x, 0.6), 0.75, 0.65, facecolor="#c8e6c9",
                           edgecolor="#2e7d32"))
    ax.add_patch(FancyArrowPatch((x + 0.37, 1.85), (x + 0.37, 1.3),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#2e7d32", linewidth=2))
ax.text(2.6, 0.2, "each partition works alone → fast, parallel",
        ha="center", fontsize=8, color="#2e7d32")
ax.text(2.6, 2.75, "filter · select · withColumn", ha="center", fontsize=8,
        color="#546e7a")

ax.text(8.0, 3.1, "WIDE  (shuffle!)", ha="center", fontsize=10,
        fontweight="bold", color="#c62828")
tops = []
bots = []
for i in range(4):
    x = 6.1 + i * 1.0
    ax.add_patch(Rectangle((x, 1.9), 0.75, 0.65, facecolor="#ffcdd2",
                           edgecolor="#c62828"))
    ax.add_patch(Rectangle((x, 0.6), 0.75, 0.65, facecolor="#ffcdd2",
                           edgecolor="#c62828"))
    tops.append(x + 0.37)
    bots.append(x + 0.37)
for t in tops:
    for b in bots:
        ax.add_patch(FancyArrowPatch((t, 1.85), (b, 1.3), arrowstyle="-|>",
                                     mutation_scale=7, color="#ef9a9a",
                                     linewidth=0.8, alpha=0.75))
ax.text(8.0, 0.2, "rows must MOVE between partitions → slow, networked",
        ha="center", fontsize=8, color="#c62828")
ax.text(8.0, 2.75, "groupBy · join · orderBy · distinct", ha="center",
        fontsize=8, color="#546e7a")
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 3.4)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

t_n, t_w = float(res["t_narrow"]), float(res["t_wide"])
c1, c2, c3 = st.columns(3)
c1.metric("narrow: filter + count", f"{t_n:.2f} s")
c2.metric("wide: groupBy + count", f"{t_w:.2f} s")
c3.metric("the shuffle tax", f"{t_w / t_n:.1f}× slower")

st.markdown(
    f"""
- **Narrow** — a partition can compute its answer using **only its own rows**.
  Filtering doesn't need to know what's on another machine. No data moves.
- **Wide** — the answer needs rows from **everywhere**. To group by `striker`,
  every delivery by Kohli must end up on the *same* machine — wherever it started.
  That physical movement of data is the **shuffle**, and it is the most expensive
  thing Spark does: data gets serialised, written to disk, sent over the network,
  and read back.

On our data the shuffle costs **{t_w / t_n:.1f}×**. On a real cluster it's the
difference between minutes and hours.

**The practical skill this gives you** — and it is a genuinely employable one:
when a Spark job is slow, open the plan, find the **`Exchange`** operators
(Spark's word for a shuffle), and ask whether each one is necessary. Half of all
Spark tuning is *"can I avoid this shuffle?"*
"""
)

c1, c2 = st.columns(2)
c1.markdown("**Narrow plan** (`filter`) — no `Exchange`:")
c1.code(res["narrow_plan"], language="text")
c2.markdown("**Wide plan** (`groupBy`) — spot the `Exchange`:")
c2.code(res["wide_plan"], language="text")

st.info(
    """
**Three shuffle-avoidance tricks worth knowing by name:**

1. **Broadcast the small side of a join** (`F.broadcast(small_df)`): sends a copy
   of the small table to every executor, so no rows have to move. Turns a wide
   join into a narrow one. The single highest-value trick in the list.
2. **Filter *before* you shuffle**, never after. Shuffling a billion rows and
   then keeping a thousand is a tragedy; Catalyst usually fixes this for you, but
   don't rely on it.
3. **Set `spark.sql.shuffle.partitions` sensibly** (default is 200). Too high on
   a laptop, often too low on a big cluster.
"""
)

show_example(
    experiments.SNIPPETS["s6_partitions"],
    """
- `df.rdd.getNumPartitions()` — how many chunks your data is in *right now*. The first thing to check when a job isn't using your cores.
- `F.spark_partition_id()` — a special column giving each row's partition number. Grouping by it is the standard way to check for **skew**: if one partition holds 90% of the rows, one poor core does 90% of the work and everyone waits for it. Skew is the top cause of mysteriously slow Spark jobs in the real world.
- `.explain()` on the narrow plan — you'll see the scan and the filter, and **no `Exchange`**. The work streams straight through.
- `.explain()` on the wide plan — **`Exchange hashpartitioning(striker, ...)`** appears. That is Spark saying *"I am about to move every row so that identical strikers land together."* Learning to spot that word in a plan is a real, practical Spark skill.
- `.repartition(n)` (a full shuffle) vs `.coalesce(n)` (merges partitions **without** a shuffle, but can only *reduce* the count). Use `coalesce` when you just want fewer output files; use `repartition` when you need balance.
""",
    key="s6_partitions",
    heavy=True,
    est="~30 s",
)

guided_sandbox(
    key="s6",
    heavy=True,
    defer=True,
    est="~90 s",
    steps="""
1. **Step 1** — print `df.rdd.getNumPartitions()`, then use
   `F.spark_partition_id()` to count the rows in each partition. Is the data
   balanced? (With one partition, the question answers itself.)
2. **Step 2** — `repartition(8)` the DataFrame and re-check the per-partition
   counts. Are they now even? This is what a shuffle buys you.
3. **Step 3** — time a `groupBy("striker").count()` at 1, 4 and 8 partitions.
   Do more partitions actually help on this dataset? Explain the result to
   yourself (remember: repartitioning *is itself* a shuffle).
4. **Step 4 (stretch)** — run `.explain()` on a `filter` and on a `groupBy`.
   Find the `Exchange` in one and its absence in the other. Then try
   `.repartition(8).filter(...)` — did adding a shuffle to a narrow operation
   make it faster or slower?
""",
    setup_code='''import time
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)
print("partitions as read:", df.rdd.getNumPartitions())

# Step 1: rows per partition


# Step 2: repartition(8) and re-check the balance


# Step 3: time groupBy at 1, 4, 8 partitions


# Step 4 (stretch): find the Exchange in the wide plan


spark.stop()
''',
    solution_code='''import time
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

print("partitions as read:", df.rdd.getNumPartitions())
print("rows per partition:")
(df.withColumn("pid", F.spark_partition_id())
   .groupBy("pid").count().orderBy("pid").show())

df8 = df.repartition(8)
print("after repartition(8):", df8.rdd.getNumPartitions())
(df8.withColumn("pid", F.spark_partition_id())
    .groupBy("pid").count().orderBy("pid").show())
print("now evenly balanced - but that balance COST a full shuffle.\\n")

for n in [1, 4, 8]:
    d = df.repartition(n)
    t0 = time.time()
    d.groupBy("striker").count().count()
    print(f"{n:2d} partitions: {time.time() - t0:.2f}s")
print("more partitions is NOT automatically faster: repartition() is itself")
print("a shuffle, and each partition adds scheduling overhead.\\n")

print("--- NARROW (filter): no Exchange ---")
df.filter(F.col("runs_off_bat") >= 4).explain()
print("--- WIDE (groupBy): look for 'Exchange hashpartitioning' ---")
df.groupBy("striker").agg(F.sum("runs_off_bat")).explain()

spark.stop()''',
)
