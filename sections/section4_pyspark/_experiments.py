"""The Spark computations behind Section 4, run once by build_artifacts.py.

Every output the pages show — execution plans, timings, result tables, the MLlib
model's scores — is captured here from a **real SparkSession** doing real work.
The pages then display those captured results instantly, and offer a
"▶ Start Spark and run it live" button.

That matters more here than anywhere else in the course: a SparkSession takes
~15 seconds to boot a JVM, and you should not pay that toll merely for opening
a page.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout

import pandas as pd

from utils.spark import DELIVERIES, INNINGS, build_session


def _capture(fn) -> str:
    """Run something that prints (like df.show()) and capture the text."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ------------------------------------------------------------------ s1
def scale_experiment() -> dict:
    """Why Spark? Measure what pandas actually costs on 1.35M rows."""
    import numpy as np

    t0 = time.time()
    pdf = pd.read_parquet(DELIVERIES)
    t_pandas_load = time.time() - t0

    mem_mb = float(pdf.memory_usage(deep=True).sum() / 1e6)
    disk_mb = float(DELIVERIES.stat().st_size / 1e6)

    t0 = time.time()
    pandas_result = (pdf.groupby("batting_team")["runs_off_bat"].sum()
                     .sort_values(ascending=False).head(8))
    t_pandas_agg = time.time() - t0

    spark = build_session()
    t0 = time.time()
    sdf = spark.read.parquet(str(DELIVERIES))
    n = sdf.count()                       # forces a real read
    t_spark_load = time.time() - t0

    from pyspark.sql import functions as F
    t0 = time.time()
    spark_result = (sdf.groupBy("batting_team")
                    .agg(F.sum("runs_off_bat").alias("runs"))
                    .orderBy(F.col("runs").desc()).limit(8).toPandas())
    t_spark_agg = time.time() - t0

    n_partitions = sdf.rdd.getNumPartitions()
    n_cores = spark.sparkContext.defaultParallelism
    spark.stop()

    return {
        "n_rows": int(n),
        "disk_mb": disk_mb,
        "pandas_mem_mb": mem_mb,
        "blowup": mem_mb / disk_mb,
        "t_pandas_load": t_pandas_load,
        "t_pandas_agg": t_pandas_agg,
        "t_spark_load": t_spark_load,
        "t_spark_agg": t_spark_agg,
        "n_partitions": int(n_partitions),
        "n_cores": int(n_cores),
        "pandas_result_csv": pandas_result.reset_index().to_csv(index=False),
        "spark_result_csv": spark_result.to_csv(index=False),
    }


# ------------------------------------------------------------------ s3
def lazy_experiment() -> dict:
    """Lazy evaluation: build a plan, then trigger it. Capture both."""
    from pyspark.sql import functions as F

    spark = build_session()
    sdf = spark.read.parquet(str(DELIVERIES))

    # --- each transformation is INSTANT: nothing is computed ---
    steps = []
    t0 = time.time()
    step1 = sdf.filter(F.col("year") >= 2015)
    steps.append(("filter(year >= 2015)", time.time() - t0))

    t0 = time.time()
    step2 = step1.filter(F.col("is_wide") == 0)
    steps.append(("filter(is_wide == 0)", time.time() - t0))

    t0 = time.time()
    step3 = step2.withColumn("is_boundary",
                             (F.col("runs_off_bat") >= 4).cast("int"))
    steps.append(("withColumn(is_boundary)", time.time() - t0))

    t0 = time.time()
    step4 = (step3.groupBy("batting_team")
             .agg(F.sum("is_boundary").alias("boundaries"),
                  F.count("*").alias("balls")))
    steps.append(("groupBy + agg", time.time() - t0))

    t0 = time.time()
    plan = step4.orderBy(F.col("boundaries").desc())
    steps.append(("orderBy", time.time() - t0))

    t_transform_total = sum(t for _, t in steps)

    # --- now an ACTION: this is what actually costs anything ---
    t0 = time.time()
    result = plan.limit(6).toPandas()
    t_action = time.time() - t0

    explain_text = _capture(lambda: plan.explain(extended=False))
    explain_formatted = _capture(lambda: plan.explain(mode="formatted"))

    spark.stop()

    return {
        "steps_csv": pd.DataFrame(
            [{"transformation": s, "seconds": t} for s, t in steps]
        ).to_csv(index=False),
        "t_transform_total": t_transform_total,
        "t_action": t_action,
        "explain": explain_text,
        "explain_formatted": explain_formatted,
        "result_csv": result.to_csv(index=False),
    }


# ------------------------------------------------------------------ s4/s5
def operations_experiment() -> dict:
    """Core DataFrame ops + the identical query in SQL — captured side by side."""
    from pyspark.sql import functions as F

    spark = build_session()
    sdf = spark.read.parquet(str(DELIVERIES))
    out = {}

    out["schema"] = _capture(sdf.printSchema)
    out["show"] = _capture(lambda: sdf.select(
        "match_id", "innings", "over", "striker", "bowler", "runs_off_bat",
        "is_wicket").show(8, truncate=False))

    # a small, readable pipeline of the core operations
    result = (sdf
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
              .orderBy(F.col("strike_rate").desc()))
    out["dsl_result"] = _capture(lambda: result.show(8, truncate=False))
    out["dsl_result_csv"] = result.limit(10).toPandas().to_csv(index=False)

    # the SAME query, in SQL
    sdf.createOrReplaceTempView("deliveries")
    sql = """
        SELECT striker,
               COUNT(*)                                  AS balls,
               SUM(runs_off_bat)                         AS runs,
               SUM(CASE WHEN runs_off_bat >= 4 THEN 1 ELSE 0 END) AS boundaries,
               SUM(is_wicket)                            AS dismissals,
               ROUND(100 * SUM(runs_off_bat) / COUNT(*), 1) AS strike_rate
        FROM deliveries
        WHERE year >= 2020 AND is_wide = 0
        GROUP BY striker
        HAVING COUNT(*) >= 500
        ORDER BY strike_rate DESC
    """
    sql_result = spark.sql(sql)
    out["sql_query"] = sql.strip()
    out["sql_result"] = _capture(lambda: sql_result.show(8, truncate=False))

    # do the two agree? (a real check, not a claim)
    a = result.limit(10).toPandas()
    b = sql_result.limit(10).toPandas()
    out["dsl_sql_identical"] = bool(a.equals(b))

    # a JOIN: bring each delivery's venue country in
    venues = spark.createDataFrame(
        [("India", "Asia"), ("Australia", "Oceania"), ("England", "Europe"),
         ("Pakistan", "Asia"), ("New Zealand", "Oceania"),
         ("South Africa", "Africa"), ("Sri Lanka", "Asia"),
         ("Bangladesh", "Asia"), ("West Indies", "Americas"),
         ("Afghanistan", "Asia")],
        ["batting_team", "region"])
    joined = (sdf.filter(F.col("year") >= 2020)
              .join(venues, on="batting_team", how="inner")
              .groupBy("region")
              .agg(F.sum("runs_off_bat").alias("runs"),
                   F.countDistinct("match_id").alias("matches"))
              .orderBy(F.col("runs").desc()))
    out["join_result"] = _capture(lambda: joined.show(truncate=False))

    spark.stop()
    return out


# ------------------------------------------------------------------ s6
def partition_experiment() -> dict:
    """Partitioning and the cost of a shuffle — measured, not asserted."""
    from pyspark.sql import functions as F

    spark = build_session()
    sdf = spark.read.parquet(str(DELIVERIES))

    default_parts = sdf.rdd.getNumPartitions()

    # how many rows are in each partition? (the thing that decides balance)
    counts = (sdf.withColumn("pid", F.spark_partition_id())
              .groupBy("pid").count().orderBy("pid").toPandas())

    rows = []
    for n_parts in [1, 2, 4, 8, 16]:
        d = sdf.repartition(n_parts)
        t0 = time.time()
        d.groupBy("batting_team").agg(F.sum("runs_off_bat")).count()
        rows.append({"partitions": n_parts, "seconds": time.time() - t0})

    # narrow vs wide: a filter needs no shuffle; a groupBy does
    t0 = time.time()
    sdf.filter(F.col("runs_off_bat") >= 4).count()
    t_narrow = time.time() - t0

    t0 = time.time()
    sdf.groupBy("striker").agg(F.sum("runs_off_bat")).count()
    t_wide = time.time() - t0

    narrow_plan = _capture(
        lambda: sdf.filter(F.col("runs_off_bat") >= 4).explain(False))
    wide_plan = _capture(
        lambda: sdf.groupBy("striker").agg(F.sum("runs_off_bat")).explain(False))

    spark.stop()
    return {
        "default_partitions": int(default_parts),
        "partition_counts_csv": counts.to_csv(index=False),
        "timings_csv": pd.DataFrame(rows).to_csv(index=False),
        "t_narrow": t_narrow,
        "t_wide": t_wide,
        "narrow_plan": narrow_plan,
        "wide_plan": wide_plan,
    }


# ------------------------------------------------------------------ s7
def mllib_experiment() -> dict:
    """A full MLlib pipeline: will this delivery be hit for a boundary?

    Run twice — once on the raw columns, once after engineering features **in
    Spark** — because the honest lesson (as in Section 7) is that the features
    move the needle more than the model does.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import (LogisticRegression,
                                           RandomForestClassifier)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import (OneHotEncoder, StandardScaler,
                                    StringIndexer, VectorAssembler)
    from pyspark.sql import functions as F

    spark = build_session()
    sdf = (spark.read.parquet(str(DELIVERIES))
           .filter(F.col("is_wide") == 0)
           .filter(F.col("year") >= 2015)
           .withColumn("label", (F.col("runs_off_bat") >= 4).cast("double")))

    train, test = sdf.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    n_train, n_test = train.count(), test.count()
    base_rate = float(train.select(F.mean("label")).first()[0])

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    rows = []

    # ---------- A. the naive feature set: raw columns only ----------
    naive_stages = [
        StringIndexer(inputCols=["batting_team", "bowling_team"],
                      outputCols=["bat_idx", "bowl_idx"], handleInvalid="keep"),
        OneHotEncoder(inputCols=["bat_idx", "bowl_idx"],
                      outputCols=["bat_oh", "bowl_oh"]),
        VectorAssembler(inputCols=["over", "innings", "bat_oh", "bowl_oh"],
                        outputCol="features_raw"),
        StandardScaler(inputCol="features_raw", outputCol="features"),
    ]
    for name, clf in [("LogisticRegression", LogisticRegression(maxIter=20)),
                      ("RandomForest",
                       RandomForestClassifier(numTrees=20, maxDepth=6, seed=42))]:
        t0 = time.time()
        model = Pipeline(stages=naive_stages + [clf]).fit(train)
        t_fit = time.time() - t0
        auc = evaluator.evaluate(model.transform(test))
        rows.append({"features": "raw columns", "model": name,
                     "auc": float(auc), "fit_seconds": t_fit})

    # ---------- B. features engineered IN SPARK ----------
    # each player's historical boundary rate, learned from the TRAINING split
    # only (target encoding — compute on train, apply to both)
    sk = (train.groupBy("striker")
          .agg(F.avg("label").alias("striker_rate"), F.count("*").alias("n"))
          .filter(F.col("n") >= 200).select("striker", "striker_rate"))
    bw = (train.groupBy("bowler")
          .agg(F.avg("label").alias("bowler_rate"), F.count("*").alias("n"))
          .filter(F.col("n") >= 200).select("bowler", "bowler_rate"))

    def enrich(df):
        return (df.join(sk, "striker", "left")
                .join(bw, "bowler", "left")
                .fillna({"striker_rate": base_rate, "bowler_rate": base_rate})
                .withColumn("is_death", (F.col("over") >= 40).cast("double"))
                .withColumn("is_powerplay", (F.col("over") < 10).cast("double")))

    tr_e, te_e = enrich(train), enrich(test)
    eng_stages = [
        VectorAssembler(
            inputCols=["over", "innings", "striker_rate", "bowler_rate",
                       "is_death", "is_powerplay"],
            outputCol="features_raw"),
        StandardScaler(inputCol="features_raw", outputCol="features"),
    ]
    for name, clf in [("LogisticRegression", LogisticRegression(maxIter=20)),
                      ("RandomForest",
                       RandomForestClassifier(numTrees=20, maxDepth=6, seed=42))]:
        t0 = time.time()
        model = Pipeline(stages=eng_stages + [clf]).fit(tr_e)
        t_fit = time.time() - t0
        auc = evaluator.evaluate(model.transform(te_e))
        rows.append({"features": "engineered in Spark", "model": name,
                     "auc": float(auc), "fit_seconds": t_fit})

    # what does VectorAssembler actually produce? show a few rows
    fitted = Pipeline(stages=eng_stages).fit(tr_e)
    features_show = _capture(lambda: fitted.transform(tr_e).select(
        "label", "over", "striker_rate", "bowler_rate", "features"
    ).show(4, truncate=55))

    # the boundary rate by over — the pattern the model is picking up
    by_over = (sdf.groupBy("over").agg(F.avg("label").alias("boundary_rate"))
               .orderBy("over").toPandas())

    spark.stop()
    return {
        "n_train": int(n_train), "n_test": int(n_test),
        "base_rate": base_rate,
        "naive_stages": ",".join(type(s).__name__ for s in naive_stages),
        "eng_stages": ",".join(type(s).__name__ for s in eng_stages),
        "results_csv": pd.DataFrame(rows).to_csv(index=False),
        "features_show": features_show,
        "by_over_csv": by_over.to_csv(index=False),
    }


# ------------------------------------------------------------------ s8
def comparison_experiment() -> dict:
    """pandas vs PySpark, same task, measured on this laptop."""
    from pyspark.sql import functions as F

    tasks = []

    pdf = pd.read_parquet(DELIVERIES)
    spark = build_session()
    sdf = spark.read.parquet(str(DELIVERIES))
    sdf.count()   # warm the read so we time the work, not the first-touch

    # 1. filter + count
    t0 = time.time()
    _ = int((pdf.query("year >= 2015 and is_wide == 0")).shape[0])
    t_p = time.time() - t0
    t0 = time.time()
    _ = sdf.filter((F.col("year") >= 2015) & (F.col("is_wide") == 0)).count()
    t_s = time.time() - t0
    tasks.append({"task": "filter + count", "pandas_s": t_p, "spark_s": t_s})

    # 2. groupBy + agg
    t0 = time.time()
    _ = pdf.groupby("striker")["runs_off_bat"].sum()
    t_p = time.time() - t0
    t0 = time.time()
    _ = sdf.groupBy("striker").agg(F.sum("runs_off_bat")).count()
    t_s = time.time() - t0
    tasks.append({"task": "groupBy + sum", "pandas_s": t_p, "spark_s": t_s})

    # 3. distinct count
    t0 = time.time()
    _ = pdf["bowler"].nunique()
    t_p = time.time() - t0
    t0 = time.time()
    _ = sdf.select("bowler").distinct().count()
    t_s = time.time() - t0
    tasks.append({"task": "count distinct bowlers", "pandas_s": t_p,
                  "spark_s": t_s})

    spark.stop()
    return {"timings_csv": pd.DataFrame(tasks).to_csv(index=False),
            "n_rows": int(len(pdf))}


# ================================================================
# The expensive teaching snippets: each boots a real SparkSession, so they
# are executed ONCE by build_artifacts.py, never on the render path.
# ================================================================

SNIPPETS: dict[str, str] = {}

SNIPPETS['s3_lazy'] = '''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

import time

# --- TRANSFORMATIONS: each returns instantly. Nothing is computed. ---
t0 = time.time()
plan = (df
        .filter(F.col("year") >= 2015)          # lazy
        .filter(F.col("is_wide") == 0)          # lazy
        .withColumn("is_boundary", (F.col("runs_off_bat") >= 4).cast("int"))
        .groupBy("batting_team")                # lazy
        .agg(F.sum("is_boundary").alias("boundaries"))
        .orderBy(F.col("boundaries").desc()))   # lazy
print(f"building the whole plan took {1000 * (time.time() - t0):.0f} ms")
print("...and read exactly ZERO rows.\\n")

# --- ACTION: now it all runs ---
t0 = time.time()
result = plan.limit(5).toPandas()
print(f"the action took {time.time() - t0:.2f} s")
print(result.to_string(index=False))

spark.stop()'''

SNIPPETS['s4_pipeline'] = '''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

result = (df
    .filter(F.col("year") >= 2020)              # 1. keep recent deliveries
    .filter(F.col("is_wide") == 0)              #    (a wide isn't a ball faced)
    .withColumn("is_boundary",                  # 2. derive a new column
                (F.col("runs_off_bat") >= 4).cast("int"))
    .groupBy("striker")                         # 3. one group per batter
    .agg(F.count("*").alias("balls"),           # 4. several aggregates at once
         F.sum("runs_off_bat").alias("runs"),
         F.sum("is_boundary").alias("boundaries"),
         F.sum("is_wicket").alias("dismissals"))
    .filter(F.col("balls") >= 500)              # 5. drop small samples
    .withColumn("strike_rate",                  # 6. a derived metric
                F.round(100 * F.col("runs") / F.col("balls"), 1))
    .orderBy(F.col("strike_rate").desc()))      # 7. sort

result.show(8, truncate=False)
spark.stop()'''

SNIPPETS['s4_join'] = '''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

# a tiny lookup table, created from a Python list
regions = spark.createDataFrame(
    [("India", "Asia"), ("Australia", "Oceania"), ("England", "Europe"),
     ("Pakistan", "Asia"), ("New Zealand", "Oceania"),
     ("South Africa", "Africa"), ("Sri Lanka", "Asia")],
    ["batting_team", "region"])

joined = (df
    .filter(F.col("year") >= 2020)
    .join(regions, on="batting_team", how="inner")   # inner / left / outer
    .groupBy("region")
    .agg(F.sum("runs_off_bat").alias("runs"),
         F.countDistinct("match_id").alias("matches"))
    .orderBy(F.col("runs").desc()))

joined.show(truncate=False)
spark.stop()'''

SNIPPETS['s5_sql'] = '''from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

# 1. give the DataFrame a NAME that SQL can see
df.createOrReplaceTempView("deliveries")

# 2. …now just write SQL
top = spark.sql("""
    SELECT batting_team,
           COUNT(*)          AS balls,
           SUM(runs_off_bat) AS runs,
           ROUND(100.0 * SUM(runs_off_bat) / COUNT(*), 1) AS strike_rate
    FROM deliveries
    WHERE year >= 2020 AND is_wide = 0
    GROUP BY batting_team
    HAVING COUNT(*) >= 5000
    ORDER BY strike_rate DESC
""")

top.show(6, truncate=False)

# 3. the result is a normal DataFrame - keep chaining Python on it if you like
print("type:", type(top).__name__)
print("columns:", top.columns)

spark.stop()'''

SNIPPETS['s6_partitions'] = '''from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = load_deliveries(spark)

print("partitions as read:", df.rdd.getNumPartitions())

# how many rows are in each partition? (badly balanced = badly parallel)
(df.withColumn("partition_id", F.spark_partition_id())
   .groupBy("partition_id").count().orderBy("partition_id").show())

# NARROW: no shuffle needed
narrow = df.filter(F.col("runs_off_bat") >= 4)
print("--- narrow plan (no Exchange) ---")
narrow.explain()

# WIDE: rows must move so that each striker's balls land together
wide = df.groupBy("striker").agg(F.sum("runs_off_bat"))
print("--- wide plan (look for 'Exchange hashpartitioning') ---")
wide.explain()

spark.stop()'''

SNIPPETS['s7_pipeline'] = '''from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = (load_deliveries(spark)
      .filter(F.col("is_wide") == 0)
      .filter(F.col("year") >= 2015)
      .withColumn("label", (F.col("runs_off_bat") >= 4).cast("double")))

train, test = df.randomSplit([0.8, 0.2], seed=42)
train.cache()          # we touch `train` many times -> keep it in memory

# --- feature engineering, IN SPARK, on the TRAINING split only ---
base = train.select(F.avg("label")).first()[0]
striker_rate = (train.groupBy("striker")
                .agg(F.avg("label").alias("striker_rate"),
                     F.count("*").alias("n"))
                .filter(F.col("n") >= 200)          # ignore tiny samples
                .select("striker", "striker_rate"))

def enrich(d):
    return (d.join(striker_rate, "striker", "left")
             .fillna({"striker_rate": base})        # unseen batter -> the average
             .withColumn("is_death", (F.col("over") >= 40).cast("double"))
             .withColumn("is_powerplay", (F.col("over") < 10).cast("double")))

train, test = enrich(train), enrich(test)

# --- the pipeline ---
pipeline = Pipeline(stages=[
    VectorAssembler(                                    # Transformer
        inputCols=["over", "innings", "striker_rate", "is_death",
                   "is_powerplay"],
        outputCol="features_raw"),
    StandardScaler(inputCol="features_raw",             # Estimator
                   outputCol="features"),
    LogisticRegression(maxIter=20),                     # Estimator
])

model = pipeline.fit(train)          # ONE call fits every stage, in order
preds = model.transform(test)        # ONE call applies every stage, in order

auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(preds)
print(f"AUC: {auc:.4f}   (0.5 = random)")

preds.select("label", "striker_rate", "probability", "prediction").show(5)
spark.stop()'''

