import streamlit as st

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🗄️ Spark SQL")
st.markdown(
    """
Here is a fact that surprises people: **you can write plain SQL against a Spark
DataFrame**, and it is not a second-class citizen. It goes through the *exact
same* Catalyst optimiser and produces the *exact same* physical plan as the
Python DSL.

That means the choice between them is purely about **readability and who has to
maintain it** — not performance. Which is liberating: use whichever expresses the
query more clearly, and don't let anyone tell you SQL is "slower".

Why this matters for employability: in a real team, analysts write SQL, data
engineers write Python, and everyone has to read both. Spark lets a single job
mix them freely.
"""
)

res = artifacts.load("s4_operations")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

st.header("1 · The same query, both ways")
c1, c2 = st.columns(2)
c1.markdown("**The Python DSL**")
c1.code('''(df
  .filter(F.col("year") >= 2020)
  .filter(F.col("is_wide") == 0)
  .withColumn("is_boundary",
      (F.col("runs_off_bat") >= 4).cast("int"))
  .groupBy("striker")
  .agg(F.count("*").alias("balls"),
       F.sum("runs_off_bat").alias("runs"),
       F.sum("is_boundary").alias("boundaries"),
       F.sum("is_wicket").alias("dismissals"))
  .filter(F.col("balls") >= 500)
  .withColumn("strike_rate",
      F.round(100 * F.col("runs") / F.col("balls"), 1))
  .orderBy(F.col("strike_rate").desc()))''', language="python")
c2.markdown("**Spark SQL**")
c2.code(res["sql_query"], language="sql")

if res["dsl_sql_identical"]:
    st.success(
        "✅ **Verified: the two produce byte-identical results.** This wasn't "
        "assumed — `build_artifacts.py` ran both and compared the output "
        "DataFrames row by row. They go through the same optimiser and end up "
        "as the same physical plan."
    )

st.markdown("**The SQL version's real output:**")
st.code(res["sql_result"], language="text")

st.header("2 · How it works: register a view, then query it")
show_example(
    experiments.SNIPPETS["s5_sql"],
    """
- `createOrReplaceTempView("deliveries")` — registers the DataFrame under a name in Spark's catalogue. It's **temporary** (it dies with your session) and it's a **view**, not a copy — no data is moved or duplicated.
- `spark.sql(\"\"\"...\"\"\")` — hand Spark a SQL string; get a DataFrame back. The triple-quoted string lets you format the SQL readably.
- `HAVING COUNT(*) >= 5000` — the SQL name for "filter the *groups*, not the rows". This is precisely the `.filter()`-after-`.groupBy()` from the previous page. If you've ever been confused about `WHERE` vs `HAVING`, that mapping is the clearest way to see it: `WHERE` filters rows *before* grouping, `HAVING` filters groups *after*.
- **`top` is an ordinary DataFrame.** You can keep chaining `.filter()`, `.join()`, `.write` on it. SQL and Python are interchangeable *mid-pipeline* — a genuinely useful thing on a mixed team.
- `spark.sql` is lazy too: the query isn't run until `.show()`.
""",
    key="s5_sql",
    heavy=True,
    est="~30 s",
)

st.info(
    """
**When to use which — a defensible answer for an interview:**

- **SQL** for anything a business analyst might want to read: joins, aggregations,
  filters. It's declarative, universally understood, and easy to review.
- **The Python DSL** when the logic is *dynamic* — building a query
  programmatically (e.g. looping over a list of columns), or chaining into
  MLlib, or when you want to unit-test each step. Building SQL by
  string-concatenation is how you get injection bugs and unreadable code.

They compile to the same thing, so **pick whichever a future reader will thank
you for.**
"""
)

guided_sandbox(
    key="s5",
    heavy=True,
    defer=True,
    est="~60 s",
    steps="""
1. **Step 1** — register the deliveries DataFrame as a temp view and write a
   SQL query for the **top 10 run-scorers since 2015** (`SELECT striker,
   SUM(runs_off_bat) ... GROUP BY ... ORDER BY ... LIMIT 10`).
2. **Step 2** — write the **identical** query using the Python DSL. Compare the
   two results with `.toPandas().equals(...)` and confirm they match exactly.
3. **Step 3** — run `.explain()` on both. Are the physical plans the same?
   (They should be — that's the point.)
4. **Step 4 (stretch)** — write a SQL query with a **window function**: for each
   batting team, rank their batters by runs scored
   (`RANK() OVER (PARTITION BY batting_team ORDER BY SUM(runs_off_bat) DESC)`).
   Window functions are where SQL really earns its keep — expressing this in the
   DSL is possible but far uglier.
""",
    setup_code='''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)
df.createOrReplaceTempView("deliveries")
print("view 'deliveries' registered; columns:", df.columns)

# Step 1: SQL - top 10 run scorers since 2015


# Step 2: the same query in the Python DSL; check they match


# Step 3: .explain() both - are the plans identical?


# Step 4 (stretch): a SQL window function - rank batters within each team


spark.stop()
''',
    solution_code='''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)
df.createOrReplaceTempView("deliveries")

sql_q = spark.sql("""
    SELECT striker, SUM(runs_off_bat) AS runs, COUNT(*) AS balls
    FROM deliveries
    WHERE year >= 2015 AND is_wide = 0
    GROUP BY striker
    ORDER BY runs DESC
    LIMIT 10
""")
print("--- SQL ---")
sql_q.show(truncate=False)

dsl_q = (df.filter((F.col("year") >= 2015) & (F.col("is_wide") == 0))
           .groupBy("striker")
           .agg(F.sum("runs_off_bat").alias("runs"),
                F.count("*").alias("balls"))
           .orderBy(F.col("runs").desc())
           .limit(10))
print("--- DSL ---")
dsl_q.show(truncate=False)

same = sql_q.toPandas().equals(dsl_q.toPandas())
print("identical results:", same)

print("\\n--- SQL plan ---")
sql_q.explain()
print("--- DSL plan ---")
dsl_q.explain()
print("same optimiser, same plan. The choice is about READABILITY.")

print("\\n--- a window function: top 3 batters per team ---")
spark.sql("""
    WITH totals AS (
        SELECT batting_team, striker, SUM(runs_off_bat) AS runs
        FROM deliveries WHERE year >= 2015
        GROUP BY batting_team, striker
    ),
    ranked AS (
        SELECT *, RANK() OVER (PARTITION BY batting_team
                               ORDER BY runs DESC) AS rk
        FROM totals
    )
    SELECT batting_team, striker, runs
    FROM ranked WHERE rk <= 3 AND batting_team IN ('India', 'Australia')
    ORDER BY batting_team, runs DESC
""").show(truncate=False)

spark.stop()''',
)
