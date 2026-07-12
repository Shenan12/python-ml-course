import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🎛️ SparkSession, and RDDs vs DataFrames")

st.header("1 · The SparkSession is your entry point to everything")
st.markdown(
    """
Every Spark program starts by creating a **SparkSession**. It is the object
through which you read data, run SQL, and submit work. In a real job you write
this once, at the top of the script, and never think about it again — but two
things about it are worth understanding properly.

**It is expensive.** Creating a SparkSession boots a **Java Virtual Machine**
and starts a small cluster's worth of machinery. It takes ~10–20 seconds. You
create **one** and reuse it for the life of your program.

**It is the driver.** Your Python code is the **driver**: it plans the work but
holds no data. The actual data lives in **executors** — separate processes that
each own some partitions. On a real cluster those are different machines; in
`local[*]` mode they're threads using your CPU cores.
"""
)

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.add_patch(Rectangle((0.4, 2.3), 2.9, 1.1, facecolor="#e3f2fd",
                       edgecolor="#1565c0", linewidth=2))
ax.text(1.85, 3.05, "YOUR PYTHON CODE", ha="center", fontsize=9,
        fontweight="bold")
ax.text(1.85, 2.6, "the DRIVER\nplans the work,\nholds no data",
        ha="center", fontsize=8)
ax.add_patch(FancyArrowPatch((3.35, 2.85), (4.35, 2.85), arrowstyle="-|>",
                             mutation_scale=16, color="#78909c", linewidth=2))
ax.text(3.85, 3.0, "py4j", ha="center", fontsize=7, color="#78909c")
ax.add_patch(Rectangle((4.4, 2.3), 2.4, 1.1, facecolor="#fff3e0",
                       edgecolor="#e65100", linewidth=2))
ax.text(5.6, 3.05, "THE JVM", ha="center", fontsize=9, fontweight="bold")
ax.text(5.6, 2.6, "Spark's engine\n(Scala/Java)", ha="center", fontsize=8)

for i in range(4):
    x = 1.0 + i * 2.1
    ax.add_patch(Rectangle((x, 0.5), 1.75, 1.15, facecolor="#e8f5e9",
                           edgecolor="#2e7d32"))
    ax.text(x + 0.87, 1.36, f"EXECUTOR {i + 1}", ha="center", fontsize=7.5,
            fontweight="bold")
    ax.text(x + 0.87, 0.95, "holds partitions\nruns your code\non ITS data",
            ha="center", fontsize=6.8)
    ax.add_patch(FancyArrowPatch((5.6, 2.25), (x + 0.87, 1.7),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#b0bec5", linewidth=1.2))
ax.text(5.0, 0.2, "in local[*] mode these are threads on your CPU cores; "
        "on a cluster they are separate machines",
        ha="center", fontsize=7.5, color="#546e7a")
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 3.6)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
**The thing to remember:** the driver never sees the data. When you call
`.collect()` or `.toPandas()`, you are **pulling every row back into the
driver's memory** — which is exactly the thing Spark exists to avoid. On a
100 GB dataset, `.collect()` will kill your driver. This is the single most
common way beginners blow up a Spark job.
"""
)

show_example(
    experiments.SNIPPETS["s2_session"],
    """
- `SparkSession.builder` — the standard construction pattern. Chain `.config(...)` calls, then finish with `.getOrCreate()`.
- `.master("local[*]")` — **the one line that decides where the work happens.** `local[*]` = this laptop, all cores. `local[2]` = pretend you have 2 machines. On a real cluster this becomes something like `yarn` or `spark://host:7077` — **and nothing else in your code changes.** That is the migration path, in full.
- `.config("spark.ui.enabled", "false")` — Spark normally starts a web dashboard on port 4040. Superb for real debugging (it shows every stage and shuffle); switched off here because we're embedded in Streamlit.
- `.config("spark.sql.shuffle.partitions", "8")` — **a genuinely important default to know.** Out of the box Spark uses **200** partitions after every shuffle. On a cluster that's sensible; on a laptop it means 200 tiny tasks with 200 lots of overhead. Setting it to roughly your core count is the standard local fix.
- `.getOrCreate()` — returns the existing session if there is one. This is why you can call it repeatedly without spawning ten JVMs.
- `setLogLevel("ERROR")` — without this, Spark buries your output in INFO logs.
""",
    key="s2_session",
    heavy=True,
    est="~20 s",
)

st.info(
    "**In this app**, the session lives in `utils/spark.py` behind "
    "`@st.cache_resource` — Streamlit's decorator for caching *objects* (as "
    "opposed to `@st.cache_data`, which caches *values*). A SparkSession is a "
    "live connection, not data, so `cache_resource` is the correct choice. It "
    "gets built once, on your first ▶ click, and reused thereafter."
)

st.header("2 · RDDs vs DataFrames — know the history, use the DataFrame")
st.markdown(
    """
You will see **RDDs** (Resilient Distributed Datasets) in older tutorials and
in interview questions. They were Spark's original API: a distributed collection
of arbitrary Python objects, which you manipulate with `map`, `filter`,
`reduce`.

**DataFrames are the modern standard, and you should use them by default.** Not
because RDDs are broken, but because of one decisive advantage:
"""
)
st.markdown(
    """
| | RDD | DataFrame |
|---|---|---|
| what it holds | arbitrary Python objects | rows with a **known schema** (named, typed columns) |
| how you use it | `.map(lambda x: ...)` | `.select()`, `.filter()`, `.groupBy()`, SQL |
| **can Spark optimise it?** | ❌ **No** — your lambda is an opaque black box | ✅ **Yes** — Catalyst rewrites your query |
| speed in PySpark | slow: every row crosses the Python↔JVM boundary | fast: the work stays in the JVM |
| when to use | rare: unstructured data, custom low-level logic | **almost always** |

**The killer point is the optimiser.** When you write a DataFrame query, Spark's
**Catalyst** optimiser sees the *whole plan* and rewrites it — reordering
filters, skipping columns you never use, pushing conditions down into the file
reader. When you write an RDD `map(lambda ...)`, Spark cannot see inside your
Python function, so it must run it exactly as written, shipping every row into a
Python process and back.

In PySpark specifically, that difference is enormous. The next page shows the
optimiser at work.
"""
)

show_example(
    '''# Both compute the same thing. Only one of them can be optimised.

# --- RDD style (the 2014 way) ---
rdd_code = """
rdd = sc.textFile("deliveries.csv")
(rdd.map(lambda line: line.split(","))
    .filter(lambda cols: int(cols[1]) >= 2015)
    .map(lambda cols: (cols[6], int(cols[10])))   # (team, runs)
    .reduceByKey(lambda a, b: a + b)
    .collect())
"""

# --- DataFrame style (what you should write) ---
df_code = """
(spark.read.parquet("deliveries.parquet")
    .filter(col("year") >= 2015)
    .groupBy("batting_team")
    .agg(sum("runs_off_bat"))
    .show())
"""

print("RDD version:")
print(rdd_code)
print("DataFrame version:")
print(df_code)
print("Same answer. But in the DataFrame version Spark KNOWS that:")
print("  - only 2 of the 15 columns are needed  -> reads just those")
print("  - the year filter can run INSIDE the parquet reader")
print("    -> rows from 2014 are never even decoded")
print("It cannot know any of that from a lambda.")''',
    """
- The RDD version treats every line as text and hand-parses it. Notice you have to know that year is column 1 and runs is column 10 — **positional, brittle, unreadable**.
- The DataFrame version names its columns and reads like a query. It is shorter, safer, and — crucially — *declarative*: you state **what** you want, not **how** to get it. That freedom is what lets Catalyst optimise.
- `sc.textFile` / `.map` / `.reduceByKey` — worth being able to recognise these in an interview, since they still come up. But if a candidate reaches for RDDs on structured data in 2026, that's a red flag.
- **The rule:** use DataFrames. Drop to RDDs only for genuinely unstructured data or custom partition-level logic — and be able to say why.
""",
)

guided_sandbox(
    key="s2",
    heavy=True,
    defer=True,
    est="~30 s",
    steps="""
1. **Step 1** — build a SparkSession with `build_session()` from
   `utils.spark`, then print `spark.version`, `spark.sparkContext.master`, and
   `spark.sparkContext.defaultParallelism` (your core count).
2. **Step 2** — make a tiny RDD: `sc.parallelize(range(1, 11))`, then use
   `.map()` and `.filter()` to square the even numbers, and `.collect()` the
   result. Note that `collect()` pulls everything back to the driver.
3. **Step 3** — do the same thing as a DataFrame:
   `spark.range(1, 11)`, then `.filter()` and `.withColumn()`. Call
   `.show()`. Compare how the two read.
4. **Step 4 (stretch)** — call `.explain()` on your DataFrame. You'll see a
   physical plan. Now try to call `.explain()` on your RDD… you can't. **That
   absence is the whole argument for DataFrames**: no plan, no optimiser.
   (Remember to `spark.stop()` at the end.)
""",
    setup_code='''from pyspark.sql import functions as F
from utils.spark import build_session

spark = build_session()
sc = spark.sparkContext
print("session ready")

# Step 1: print version, master, core count


# Step 2: an RDD - square the even numbers 1..10


# Step 3: the same thing as a DataFrame


# Step 4 (stretch): .explain() the DataFrame; note the RDD has no plan


spark.stop()
''',
    solution_code='''from pyspark.sql import functions as F
from utils.spark import build_session

spark = build_session()
sc = spark.sparkContext

print("Spark version:", spark.version)
print("master       :", sc.master)
print("cores        :", sc.defaultParallelism)

rdd = sc.parallelize(range(1, 11))
squares = rdd.filter(lambda x: x % 2 == 0).map(lambda x: x * x)
print("\\nRDD result:", squares.collect())
print("(collect() pulled every row back into the DRIVER - fine for 5 numbers,")
print(" fatal for 100 GB.)")

df = (spark.range(1, 11)
      .filter(F.col("id") % 2 == 0)
      .withColumn("square", F.col("id") ** 2))
print("\\nDataFrame result:")
df.show()

print("the DataFrame has a PLAN Spark can optimise:")
df.explain()
print("the RDD has no such thing - its lambdas are opaque to Spark.")

spark.stop()''',
)
