import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import artifacts
from utils.sandbox import guided_sandbox

st.title("🔥 Why Spark Exists")
st.markdown(
    """
This section is the **employability** one. Nearly every data-engineering and
many data-science job adverts list Spark, and the reason is a single, brutally
practical problem: **pandas loads your entire dataset into your computer's
memory.** When the data no longer fits, pandas doesn't get slow — it *dies*.

Everything you learn here uses the **same API** you'd use on a 500-machine
cluster at a bank or a streaming company. We just run it on your laptop.
"""
)

res = artifacts.load("s1_scale")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

st.header("1 · The number that explains everything")
st.markdown(
    "Our dataset is the **ball-by-ball record of every men's ODI** — the same "
    "Cricsheet source Section 7 used, but at delivery level instead of innings "
    "level. Here's what happens when pandas opens it:"
)

disk = float(res["disk_mb"])
mem = float(res["pandas_mem_mb"])
blowup = float(res["blowup"])

c1, c2, c3 = st.columns(3)
c1.metric("rows (deliveries)", f"{int(res['n_rows']):,}")
c2.metric("file size on disk", f"{disk:.1f} MB")
c3.metric("RAM used by pandas", f"{mem:.0f} MB", f"{blowup:.0f}× bigger",
          delta_color="inverse")

fig, ax = plt.subplots(figsize=(8.5, 2.2))
ax.barh(["on disk (parquet)", "in pandas (RAM)"], [disk, mem],
        color=["#2e7d32", "#c62828"], edgecolor="white")
for i, v in enumerate([disk, mem]):
    ax.text(v + mem * 0.01, i, f"{v:.1f} MB", va="center", fontsize=10)
ax.set_xlabel("megabytes")
ax.set_xlim(0, mem * 1.18)
st.pyplot(fig)
plt.close(fig)
st.caption(f"📦 {artifacts.provenance(res)} — measured on your machine.")

st.error(
    f"""
**A {disk:.1f} MB file becomes {mem:.0f} MB of RAM — a {blowup:.0f}× blow-up.**

Why? Parquet on disk is *columnar and compressed*: the string `"India"` appears
once in a dictionary and every row just points at it. The moment pandas loads
it, every one of those {int(res['n_rows']):,} rows gets its own Python string
object, with all the overhead that implies.

Now extrapolate. This is **one sport's ODI history** — genuinely small data. A
mid-sized company's clickstream is 500 GB *per day*. At that scale:

- **8 GB of RAM** on your laptop is gone in the first two minutes.
- You cannot `pd.read_csv` your way out of it. There is no flag for this.
- And a machine with 500 GB of RAM costs more than a small cluster and still
  can't be scaled past its motherboard.
"""
)

st.header("2 · What Spark does instead")
st.markdown(
    """
Spark's answer is not "be faster at the same thing". It's a different shape of
computation entirely:

1. **Split the data into partitions** — chunks that each fit comfortably in
   memory.
2. **Send the *code* to the data**, not the data to the code. Each machine
   (or, on your laptop, each **CPU core**) processes its own partitions
   independently.
3. **Combine the partial results** at the end.

The killer property: your **8 GB laptop can process a 100 GB file**, because it
never holds more than a few partitions at a time. Spark streams through the data
partition by partition, spilling to disk if it must. pandas simply cannot do
this — its model is "everything in RAM, all at once".
"""
)

fig, ax = plt.subplots(figsize=(10, 3.4))
ax.text(2.2, 3.3, "pandas", ha="center", fontsize=12, fontweight="bold",
        color="#c62828")
ax.add_patch(plt.Rectangle((0.6, 1.5), 3.2, 1.5, facecolor="#ffcdd2",
                           edgecolor="#c62828", linewidth=2))
ax.text(2.2, 2.55, "ONE process", ha="center", fontsize=9)
ax.text(2.2, 2.15, "the WHOLE dataset\nmust fit in RAM", ha="center",
        fontsize=8.5)
ax.text(2.2, 1.7, "💥 dies when it doesn't", ha="center", fontsize=9,
        color="#c62828")

ax.text(7.4, 3.3, "Spark", ha="center", fontsize=12, fontweight="bold",
        color="#2e7d32")
ax.add_patch(plt.Rectangle((5.0, 2.45), 4.8, 0.62, facecolor="#e8f5e9",
                           edgecolor="#2e7d32", linewidth=1.6))
ax.text(7.4, 2.76, "DRIVER — plans the work, holds no data", ha="center",
        fontsize=8.5)
for i in range(4):
    x = 5.05 + i * 1.22
    ax.add_patch(plt.Rectangle((x, 1.35), 1.05, 0.75, facecolor="#c8e6c9",
                               edgecolor="#2e7d32"))
    ax.text(x + 0.52, 1.9, f"core {i + 1}", ha="center", fontsize=7.5)
    ax.text(x + 0.52, 1.58, f"partition\n{i + 1}", ha="center", fontsize=7)
    ax.annotate("", xy=(x + 0.52, 2.15), xytext=(x + 0.52, 2.42),
                arrowprops=dict(arrowstyle="-|>", color="#78909c"))
ax.text(7.4, 0.95, "each core sees only its own slice → the data never has to "
        "fit all at once", ha="center", fontsize=8, color="#2e7d32")
ax.set_xlim(0, 10.2)
ax.set_ylim(0.6, 3.7)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.info(
    f"""
**Local mode — and being honest about it.** Throughout this section we run Spark
with `master("local[*]")`: the "cluster" is **your laptop**, and the "machines"
are your **{int(res['n_cores'])} CPU cores**. Spark split our data into
**{int(res['n_partitions'])} partitions** automatically.

This is a teaching setup, not a real cluster — but the **code is identical**.
Change one string from `local[*]` to a cluster URL and the same script runs on
500 machines. That is genuinely the whole migration, and it's why learning Spark
on a laptop is not a waste of time.
"""
)

st.header("3 · But is Spark actually faster? (No. And that's important.)")
st.markdown(
    "Same job — total runs by batting team — in both tools, timed on this "
    "machine:"
)
t_pl, t_pa = float(res["t_pandas_load"]), float(res["t_pandas_agg"])
t_sl, t_sa = float(res["t_spark_load"]), float(res["t_spark_agg"])

fig, ax = plt.subplots(figsize=(9, 2.6))
ax.barh(["pandas", "Spark (local)"], [t_pl, t_sl], color="#90a4ae",
        label="load the data", edgecolor="white")
ax.barh(["pandas", "Spark (local)"], [t_pa, t_sa], left=[t_pl, t_sl],
        color="#1565c0", label="groupBy + sum", edgecolor="white")
for i, (a, b) in enumerate([(t_pl, t_pa), (t_sl, t_sa)]):
    ax.text(a + b + 0.05, i, f"{a + b:.2f}s total", va="center", fontsize=9)
ax.set_xlabel("seconds")
ax.legend(fontsize=8)
st.pyplot(fig)
plt.close(fig)

winner = "pandas" if (t_pl + t_pa) < (t_sl + t_sa) else "Spark"
st.warning(
    f"""
**On this dataset, {winner} wins — and if it's pandas, that is exactly what you
should expect.**

Spark has real overhead: it boots a JVM, plans the query, serialises tasks,
distributes them, and collects results. For 1.35 million rows on one laptop,
**that overhead costs more than it saves**. pandas just... does the work.

**This is the most important thing on the page, and the thing beginners get
wrong.** Spark is not "the fast pandas". It is *the pandas that doesn't die*.
You reach for it when:

- the data **does not fit in one machine's memory** (the real reason), or
- you genuinely have a cluster and the parallelism pays for the overhead, or
- the data already lives in a distributed store (S3, HDFS, a data lake) and
  moving it to one machine is the expensive part.

If your data fits comfortably in RAM, **use pandas** and don't apologise for it.
Reaching for Spark on a 50 MB CSV marks you out as someone who has learned the
tool but not the judgement. The last page of this section gives you a decision
rule you can actually defend in an interview.
"""
)

st.header("4 · Both tools, same answer")
c1, c2 = st.columns(2)
c1.markdown("**pandas**")
c1.dataframe(pd.read_csv(io.StringIO(res["pandas_result_csv"])),
             hide_index=True, width="stretch")
c2.markdown("**Spark**")
c2.dataframe(pd.read_csv(io.StringIO(res["spark_result_csv"])),
             hide_index=True, width="stretch")
st.caption("Identical results, as they must be. The difference is never the "
           "answer — it's what happens when the data gets big.")

guided_sandbox(
    key="s1",
    heavy=True,
    defer=True,
    est="~10 s",
    steps="""
1. **Step 1** — load the deliveries parquet with pandas and print its shape and
   its true memory footprint: `df.memory_usage(deep=True).sum() / 1e6` MB.
   Compare that with the file size on disk (`DELIVERIES.stat().st_size / 1e6`).
2. **Step 2** — find where the memory actually goes: print
   `df.memory_usage(deep=True).sort_values(ascending=False) / 1e6`. Which
   columns are the hogs, and what type are they?
3. **Step 3** — shrink it. Convert the biggest string columns to pandas'
   `"category"` dtype (`df[col].astype("category")`) and re-measure. How much
   did you save? (This is the *pandas* answer to scale — and it works until it
   doesn't.)
4. **Step 4 (stretch)** — estimate the wall. If this dataset is
   `X` MB in RAM for 1.35M rows, how many rows before you exhaust an 8 GB
   laptop? At roughly what point would you *have* to reach for Spark?
""",
    setup_code='''import pandas as pd
from utils.spark import DELIVERIES

print("file:", DELIVERIES.name)
print("size on disk:", round(DELIVERIES.stat().st_size / 1e6, 1), "MB")

# Step 1: load with pandas; print shape and memory footprint


# Step 2: which columns eat the memory?


# Step 3: convert string columns to "category" and re-measure


# Step 4 (stretch): how many rows until an 8 GB laptop dies?
''',
    solution_code='''import pandas as pd
from utils.spark import DELIVERIES

disk = DELIVERIES.stat().st_size / 1e6
df = pd.read_parquet(DELIVERIES)
mem = df.memory_usage(deep=True).sum() / 1e6
print(f"shape: {df.shape}")
print(f"disk : {disk:6.1f} MB")
print(f"RAM  : {mem:6.1f} MB   ({mem / disk:.0f}x blow-up)")

print("\\nmemory by column (MB):")
by_col = (df.memory_usage(deep=True) / 1e6).sort_values(ascending=False)
for col, m in by_col.head(6).items():
    dt = df[col].dtype if col in df.columns else "-"
    print(f"  {str(col):16} {m:7.1f}  ({dt})")
print("-> the string columns dominate. Every row holds its own Python str.")

small = df.copy()
for col in ["venue", "batting_team", "bowling_team", "striker", "bowler",
            "wicket_type"]:
    small[col] = small[col].astype("category")
mem2 = small.memory_usage(deep=True).sum() / 1e6
print(f"\\nafter converting strings to 'category': {mem2:.1f} MB")
print(f"saved {mem - mem2:.0f} MB ({100 * (1 - mem2 / mem):.0f}%)")
print("categories store each distinct string ONCE + small integer codes -")
print("exactly the trick parquet uses on disk.")

per_row_kb = mem / len(df) * 1000
rows_in_8gb = 8000 / (mem / len(df)) / 1e6
print(f"\\nper row: {per_row_kb:.3f} KB")
print(f"an 8 GB laptop (say 6 GB usable) holds roughly "
      f"{6000 / (mem / len(df)) / 1e6:.1f} million rows")
print("Beyond that: category dtypes, then chunking, then Spark.")''',
)
