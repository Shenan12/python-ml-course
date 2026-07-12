import io

import pandas as pd
import streamlit as st

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🧱 Core DataFrame Operations")
st.markdown(
    """
Six verbs do most of the work in every Spark job you will ever write. If you
know pandas, you already know what they *mean* — the differences are in the
spelling, and in a few places where Spark's distributed nature shows through.
"""
)

res = artifacts.load("s4_operations")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

st.markdown(
    """
| verb | what it does | pandas equivalent |
|---|---|---|
| **`select`** | choose columns | `df[["a", "b"]]` |
| **`filter`** / `where` | keep rows matching a condition | `df[df.a > 1]` |
| **`withColumn`** | add or replace **one** column | `df["c"] = ...` |
| **`groupBy` + `agg`** | split, aggregate, combine | `df.groupby(...).agg(...)` |
| **`join`** | combine two tables on a key | `pd.merge(...)` |
| **`orderBy`** | sort | `df.sort_values(...)` |

The one that trips people up is **`withColumn`**: it adds *one* column and
returns a *new* DataFrame. Spark DataFrames are **immutable** — you never modify
one in place, you always build a new one. (Chaining twenty `withColumn` calls is
also a known performance trap; use `select` with several expressions instead when
you're adding many columns at once.)
"""
)

st.header("1 · The data")
st.code(res["schema"], language="text")
st.code(res["show"], language="text")
st.caption(f"📦 {artifacts.provenance(res)} — real Spark output on the "
           "1.35M-row deliveries table.")

st.header("2 · A real pipeline: who are the fastest scorers since 2020?")
st.markdown(
    "This chains every verb on the list. Read it top to bottom — it's a "
    "sentence."
)
show_example(
    experiments.SNIPPETS["s4_pipeline"],
    """
- `F.col("year")` — `F` is the standard alias for `pyspark.sql.functions`. `F.col("x")` refers to a *column*, not a value. You cannot write `df.year >= 2015` and expect pandas semantics; you build a **column expression** that Spark will compile into its plan.
- **`.filter()` twice** rather than one compound condition — perfectly fine, and often more readable. Catalyst will combine them anyway; the optimiser doesn't care how you wrote it.
- `.cast("int")` — the boolean becomes 0/1 so we can `sum` it. A very common idiom: `sum(condition)` = "how many rows matched".
- `.agg(...)` with several functions — each gets an `.alias(...)`. **Always alias**, or your column will be named the unreadable `sum(runs_off_bat)`.
- `.filter(F.col("balls") >= 500)` **after** the `groupBy` — this filters the *groups*, not the rows. It's the `HAVING` clause of SQL. Filtering before the groupBy would filter deliveries; here we want to drop batters with small samples.
- `.show(8, truncate=False)` — an **action**. Everything above this line was lazy. `truncate=False` stops Spark from cutting long names off at 20 characters.
""",
    key="s4_pipeline",
    heavy=True,
    est="~30 s",
)
st.markdown("**The real output:**")
st.code(res["dsl_result"], language="text")

st.header("3 · Joins")
st.markdown(
    """
Joins in Spark look like pandas merges, with one crucial difference you must
understand: **a join usually requires a shuffle** — rows with the same key must
physically move to the same machine before they can be matched. That makes joins
the most expensive thing in most Spark jobs, and the first place to look when one
is slow. (More on shuffles two pages from now.)

Here we join a small lookup table of regions onto the deliveries:
"""
)
show_example(
    experiments.SNIPPETS["s4_join"],
    """
- `spark.createDataFrame([...], ["col1", "col2"])` — builds a small DataFrame from Python data. Handy for lookups and for testing.
- `.join(regions, on="batting_team", how="inner")` — same vocabulary as `pd.merge`: `inner`, `left`, `right`, `outer`. If the key columns had *different* names you'd write `on=df.a == regions.b` instead.
- **`how="inner"` silently drops rows.** Teams not in our lookup vanish from the result. Exactly as in Section 3's pandas merge page: after any join, **check your row count**. A join that silently halves your data is the classic production bug.
- `F.countDistinct("match_id")` — distinct counts are *expensive* in a distributed system (every machine must compare with every other), which is why Spark also offers `approx_count_distinct` for when "close enough" will do at scale. Knowing that trade-off exists is worth an interview mark.
- **The optimisation to know:** when one side of a join is small (like `regions`), Spark can **broadcast** it — send a full copy to every executor and skip the shuffle entirely. It often does this automatically; you can force it with `F.broadcast(regions)`. Being able to say *"I'd broadcast the small side"* is a genuinely useful thing to have in your pocket.
""",
    key="s4_join",
    heavy=True,
    est="~30 s",
)
st.markdown("**The real output:**")
st.code(res["join_result"], language="text")

guided_sandbox(
    key="s4",
    heavy=True,
    defer=True,
    est="~60 s",
    steps="""
1. **Step 1** — find the **most economical bowlers** since 2018: group the
   deliveries by `bowler`, count balls and sum `runs_off_bat`, keep bowlers with
   at least 500 balls, compute `economy = 6 * runs / balls` (runs per over), and
   sort ascending. Show the top 10.
2. **Step 2** — add a `wickets` column to that same aggregation
   (`F.sum("is_wicket")`) and a `strike_rate_balls_per_wicket` column
   (`balls / wickets`). Who takes wickets fastest?
3. **Step 3** — join your bowler table to a small DataFrame you create yourself
   mapping 3–4 `bowling_team` names to a country/region, and count bowlers per
   region. Check whether the join dropped any rows (compare counts before and
   after).
4. **Step 4 (stretch)** — use `F.broadcast()` on your small lookup table and
   call `.explain()` on the join both with and without it. Can you spot
   `BroadcastHashJoin` versus `SortMergeJoin` in the two plans? That difference
   is worth real money on a big cluster.
""",
    setup_code='''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)
print("columns:", df.columns)

# Step 1: most economical bowlers since 2018


# Step 2: add wickets and balls-per-wicket


# Step 3: join a small lookup; check no rows were silently dropped


# Step 4 (stretch): F.broadcast() and compare the two .explain() plans


spark.stop()
''',
    solution_code='''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

bowlers = (df
    .filter(F.col("year") >= 2018)
    .filter(F.col("is_wide") == 0)
    .groupBy("bowler", "bowling_team")
    .agg(F.count("*").alias("balls"),
         F.sum("runs_off_bat").alias("runs"),
         F.sum("is_wicket").alias("wickets"))
    .filter(F.col("balls") >= 500)
    .withColumn("economy", F.round(6 * F.col("runs") / F.col("balls"), 2))
    .withColumn("balls_per_wicket",
                F.round(F.col("balls") / F.col("wickets"), 1)))

print("most economical bowlers (runs per over):")
bowlers.orderBy("economy").select(
    "bowler", "balls", "economy", "wickets").show(10, truncate=False)

print("fastest wicket-takers (fewest balls per wicket):")
bowlers.orderBy("balls_per_wicket").select(
    "bowler", "balls", "wickets", "balls_per_wicket").show(10, truncate=False)

regions = spark.createDataFrame(
    [("India", "Asia"), ("Australia", "Oceania"),
     ("England", "Europe"), ("Pakistan", "Asia")],
    ["bowling_team", "region"])

before = bowlers.count()
joined = bowlers.join(F.broadcast(regions), on="bowling_team", how="inner")
after = joined.count()
print(f"\\nbowlers before join: {before}   after inner join: {after}")
print(f"the join DROPPED {before - after} bowlers (teams not in the lookup).")
print("always check this - a silent join drop is a classic production bug.\\n")

joined.groupBy("region").agg(F.count("*").alias("bowlers")).show()

print("--- plan WITH broadcast ---")
bowlers.join(F.broadcast(regions), "bowling_team").explain()

spark.stop()''',
)
