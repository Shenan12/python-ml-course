import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import artifacts
from utils.sandbox import guided_sandbox

st.title("⚖️ pandas vs PySpark — and When to Reach for Which")

st.header("1 · The same five tasks, side by side")
st.markdown(
    "If you know pandas, this table *is* PySpark. Print it out; it's most of "
    "what you need for a first Spark job."
)

rows = [
    ("read a file",
     'pd.read_parquet("f.parquet")',
     'spark.read.parquet("f.parquet")'),
    ("look at it",
     "df.head()",
     "df.show(5)          # an ACTION"),
    ("select columns",
     'df[["a", "b"]]',
     'df.select("a", "b")'),
    ("filter rows",
     "df[df.year >= 2015]",
     'df.filter(F.col("year") >= 2015)'),
    ("new column",
     'df["x"] = df.a * 2',
     'df = df.withColumn("x", F.col("a") * 2)'),
    ("group + aggregate",
     'df.groupby("team")["runs"].sum()',
     'df.groupBy("team").agg(F.sum("runs"))'),
    ("join",
     'pd.merge(a, b, on="k", how="left")',
     'a.join(b, on="k", how="left")'),
    ("sort",
     'df.sort_values("runs", ascending=False)',
     'df.orderBy(F.col("runs").desc())'),
    ("row count",
     "len(df)",
     "df.count()          # an ACTION"),
    ("unique values",
     "df.bowler.nunique()",
     'df.select("bowler").distinct().count()'),
]
st.dataframe(
    pd.DataFrame(rows, columns=["task", "pandas", "PySpark"]),
    hide_index=True, width="stretch")

st.markdown(
    """
**The four differences that actually bite:**

1. **Immutability.** `df["x"] = ...` doesn't exist in Spark. You must *reassign*:
   `df = df.withColumn("x", ...)`. Spark DataFrames never change in place.
2. **Laziness.** `df.filter(...)` computes nothing. Only actions (`show`,
   `count`, `collect`, `write`) do work.
3. **No index.** Spark has no row index and no `.loc` / `.iloc`. There is no
   "row 5" — rows live on different machines and have no inherent order. If you
   want order, you must `orderBy`.
4. **`.toPandas()` and `.collect()` are dangerous.** They pull the entire result
   into the driver's memory. Fine for a summary table; fatal for a billion rows.
"""
)

st.header("2 · Which is actually faster? (Measured, on your laptop)")
res = artifacts.load("s8_comparison")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()
t = pd.read_csv(io.StringIO(res["timings_csv"]))

fig, ax = plt.subplots(figsize=(9, 3.0))
x = np.arange(len(t))
w = 0.36
ax.bar(x - w / 2, t["pandas_s"], w, label="pandas", color="#2e7d32",
       edgecolor="white")
ax.bar(x + w / 2, t["spark_s"], w, label="Spark (local)", color="#c62828",
       edgecolor="white")
for i, (p, s) in enumerate(zip(t["pandas_s"], t["spark_s"])):
    ax.text(i - w / 2, p + 0.01, f"{p:.2f}s", ha="center", fontsize=8)
    ax.text(i + w / 2, s + 0.01, f"{s:.2f}s", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(t["task"], fontsize=8.5)
ax.set_ylabel("seconds")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
ax.set_title(f"same tasks, {int(res['n_rows']):,} rows, one laptop", fontsize=10)
st.pyplot(fig)
plt.close(fig)
st.dataframe(t.round(3), hide_index=True, width="stretch")

st.error(
    f"""
**pandas wins every single task.** And it should — this is the honest headline of
the whole section.

At 1.35 million rows on one machine, Spark's overhead (planning, serialising
tasks, scheduling, shuffling) costs more than its parallelism saves. pandas just
reaches into RAM and does the work.

**If you take one thing from this section into an interview, take this:**

> *"Spark isn't a faster pandas — it's the pandas that doesn't run out of memory.
> On a laptop-sized dataset, pandas will beat it, and I'd use pandas. Spark earns
> its overhead when the data doesn't fit on one machine, or when it already lives
> distributed."*

That answer will impress far more than reciting that Spark is "fast" and
"distributed". Anyone can memorise a feature list. Judgement is rarer.
"""
)

st.header("3 · The decision rule")
st.markdown(
    """
Here is the flow chart I'd actually use, and would defend:
"""
)

fig, ax = plt.subplots(figsize=(10, 4.2))


def box(x, y, w, h, text, face, fs=8.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face,
                               edgecolor="#37474f", linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#546e7a",
                                linewidth=1.5))
    if label:
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2, label, fontsize=8,
                color="#546e7a", fontweight="bold")


box(3.4, 3.4, 3.4, 0.65, "Does the data fit comfortably\nin one machine's RAM?",
    "#e3f2fd")
arrow(4.3, 3.35, 2.2, 2.65, "YES")
arrow(5.9, 3.35, 7.9, 2.65, "NO")

box(0.7, 2.0, 3.0, 0.62, "Use pandas. Seriously.", "#c8e6c9")
box(1.0, 1.1, 2.4, 0.62, "It's simpler, faster,\nand better tooled.", "#f1f8e9",
    fs=7.5)

box(6.6, 2.0, 3.0, 0.62, "Can you shrink it?", "#fff3e0")
arrow(7.4, 1.95, 6.6, 1.3, "YES")
arrow(8.9, 1.95, 9.4, 1.3, "NO")
box(4.9, 0.55, 3.3, 0.7,
    "sample · pick columns ·\n'category' dtypes · chunk it\n→ still pandas",
    "#f1f8e9", fs=7.5)
box(8.6, 0.55, 2.2, 0.7, "→ USE SPARK", "#ffcdd2", fs=9.5)
ax.set_xlim(0, 11)
ax.set_ylim(0.3, 4.3)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
### The genuine reasons a real job reaches for Spark

1. **The data doesn't fit in memory.** The only *unavoidable* reason. 500 GB of
   logs cannot be `pd.read_csv`'d, at any price.
2. **The data already lives distributed** — in S3, HDFS, a data lake, a Delta
   table. Pulling 2 TB onto one machine just to use pandas is absurd; send the
   code to the data instead.
3. **There is a real cluster**, and the parallelism is genuinely paid for.
4. **The pipeline must scale.** Today it's 10 GB and pandas copes. Next year it's
   10 TB. Writing it in Spark now means the only thing that changes is the
   cluster size.
5. **The team already runs Spark.** Fitting the stack you're hired into is a
   legitimate engineering reason, and pretending otherwise is naive.

### The bad reasons (which get people caught out)

- *"Spark is faster."* It isn't, on small data. You have now measured this.
- *"Big data is the modern way."* Most companies' data fits on a laptop. It is a
  well-known industry embarrassment that a great deal of Spark runs on datasets a
  MacBook could handle.
- *"I want it on my CV."* Fine motive for *learning* it — as you are. Terrible
  motive for *choosing* it on a real project.

### The middle ground worth knowing exists

Before you jump from pandas to Spark, there is a whole tier of tools for
"bigger than RAM, smaller than a cluster": **Polars** and **DuckDB** (both
astonishingly fast on one machine), **Dask** (parallel pandas). Knowing that this
tier exists — and that Spark is not the automatic next step above pandas — is
exactly the kind of judgement that separates someone who has *learned* a tool
from someone who can *choose* one.
"""
)

st.success(
    """
**What you can now honestly claim on a CV:** you can create a SparkSession,
read distributed files, use the DataFrame API and Spark SQL, explain lazy
evaluation and read a physical plan, identify shuffles and know how to avoid
them, build an MLlib Pipeline with Transformers and Estimators, and — most
importantly — **explain when Spark is the wrong tool.**

That last one is what a good interviewer is actually probing for.
"""
)

guided_sandbox(
    key="s8",
    heavy=True,
    defer=True,
    est="~60 s",
    steps="""
1. **Step 1** — pick any question about the cricket data ("who scored most runs
   at the death?", "which bowler concedes fewest boundaries?") and answer it
   **twice**: once in pandas, once in PySpark. Time both.
2. **Step 2** — check the two answers agree (compare the resulting DataFrames).
   If they don't, one of your two pipelines has a bug — a very good habit for
   verifying a migration.
3. **Step 3** — estimate the crossover. Your pandas version used roughly
   `df.memory_usage(deep=True).sum()` bytes for 1.35M rows. At what row count
   would it exhaust 8 GB? That's roughly where Spark starts to earn its keep.
4. **Step 4 (stretch)** — try the same query in **Polars**
   (`pip install polars`, then `pl.read_parquet(...)`). It's a single-machine
   tool that is usually *faster than both*. Now you can name three tools and say
   precisely when each one wins — which is the whole point of this page.
""",
    setup_code='''import time
import pandas as pd
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries, DELIVERIES

pdf = pd.read_parquet(DELIVERIES)
spark = build_session()
sdf = load_deliveries(spark)
sdf.count()          # warm it up so we time the work, not the first read
print("both loaded:", len(pdf), "rows")

# Step 1: answer the SAME question in pandas and in Spark; time both


# Step 2: check the answers agree


# Step 3: at what row count does pandas exhaust 8 GB?


# Step 4 (stretch): try Polars, if installed


spark.stop()
''',
    solution_code='''import time
import pandas as pd
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries, DELIVERIES

pdf = pd.read_parquet(DELIVERIES)
spark = build_session()
sdf = load_deliveries(spark)
sdf.count()

# QUESTION: top 5 run-scorers in the death overs (40+) since 2015
t0 = time.time()
p_ans = (pdf[(pdf.year >= 2015) & (pdf.over >= 40) & (pdf.is_wide == 0)]
         .groupby("striker")["runs_off_bat"].sum()
         .sort_values(ascending=False).head(5))
t_pandas = time.time() - t0

t0 = time.time()
s_ans = (sdf.filter((F.col("year") >= 2015) & (F.col("over") >= 40)
                    & (F.col("is_wide") == 0))
         .groupBy("striker").agg(F.sum("runs_off_bat").alias("runs"))
         .orderBy(F.col("runs").desc()).limit(5).toPandas())
t_spark = time.time() - t0

print(f"pandas: {t_pandas:.3f}s")
print(p_ans.to_string())
print(f"\\nSpark:  {t_spark:.3f}s")
print(s_ans.to_string(index=False))

agree = list(p_ans.index) == list(s_ans.striker)
print(f"\\nsame answer: {agree}")
print(f"pandas was {t_spark / t_pandas:.1f}x faster on this laptop-sized data.")

mem_gb = pdf.memory_usage(deep=True).sum() / 1e9
per_row = mem_gb / len(pdf)
print(f"\\npandas used {mem_gb:.2f} GB for {len(pdf):,} rows")
print(f"an 8 GB laptop (~6 GB usable) would die at roughly "
      f"{6 / per_row / 1e6:.0f} million rows")
print("THAT is the crossover. Below it: pandas. Above it: Spark (or Polars).")

spark.stop()''',
)
