import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from sections.section4_pyspark import _experiments as experiments
from utils import artifacts
from utils.sandbox import guided_sandbox, show_example

st.title("🤖 MLlib: Machine Learning at Scale")
st.markdown(
    """
**MLlib** is Spark's machine-learning library. If you know scikit-learn — and
after Section 3, you do — then you already know 80% of it, because MLlib
deliberately copies scikit-learn's design. The three concepts:

| concept | what it is | scikit-learn analogue |
|---|---|---|
| **Transformer** | has `.transform()`. Turns one DataFrame into another. | anything with `.transform()` |
| **Estimator** | has `.fit()`. Learns from data and **returns a Transformer**. | anything with `.fit()` |
| **Pipeline** | chains stages so they run as one unit | `sklearn.pipeline.Pipeline` |

The distinction between Transformer and Estimator is the one to get straight,
and it's cleaner in MLlib than in sklearn:

- A `StandardScaler` is an **Estimator** — it must *learn* the mean and standard
  deviation from your training data. Calling `.fit()` gives you a
  `StandardScalerModel`, which **is** a Transformer.
- A fitted model is *always* a Transformer: it takes a DataFrame of features and
  adds a `prediction` column.

**And crucially: a `Pipeline` is an Estimator, so `pipeline.fit(train)` returns a
`PipelineModel` — one object that carries every fitted step.** That single object
is what you save, ship to production, and apply to new data. It is why the
pipeline abstraction is not decoration; it's how you avoid the classic bug of
scaling your test set with the *test set's* mean.
"""
)

res = artifacts.load("s7_mllib")
if res is None:
    st.error("Run `python build_artifacts.py --only spark` first.")
    st.stop()

st.header("1 · The task")
st.markdown(
    f"""
**Will this delivery be hit for a boundary (4 or 6)?** One row per ball, a
binary label. It's a deliberately simple problem on a genuinely large table —
**{int(res['n_train']):,} training rows**, which is the point: this would be
uncomfortable in pandas and is routine in Spark.

Only **{float(res['base_rate']):.1%}** of deliveries go to the boundary, so a
model that always guesses "no" is right ~90% of the time. **This is why we score
with AUC, not accuracy** — exactly the imbalanced-classes lesson from Section 3's
evaluation page.
"""
)

by_over = pd.read_csv(io.StringIO(res["by_over_csv"]))
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(by_over["over"], by_over["boundary_rate"] * 100, color="#c62828",
        linewidth=2.2, marker="o", markersize=3)
ax.axvspan(0, 10, color="#bbdefb", alpha=0.5)
ax.axvspan(40, 50, color="#ffcdd2", alpha=0.5)
ax.text(0.10, 0.06, "powerplay\n(field up)", ha="center", fontsize=8,
        transform=ax.transAxes)
ax.text(0.90, 0.06, "death overs\n(all-out attack)", ha="center", fontsize=8,
        transform=ax.transAxes)
ax.set_xlabel("over")
ax.set_ylabel("% of balls hit for 4 or 6")
ax.grid(alpha=0.25)
ax.set_title("the signal the model has to find (real data)", fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Boundary rate is high in the powerplay, dips in the middle overs, "
           "then explodes at the death. Any useful model must capture this.")

st.header("2 · The pipeline")
fig, ax = plt.subplots(figsize=(11, 2.2))
stages = ["raw\nDataFrame", "VectorAssembler\n(bundle columns\ninto one vector)",
          "StandardScaler\n(Estimator:\nlearns μ, σ)",
          "LogisticRegression\n(Estimator:\nlearns weights)",
          "predictions"]
colours = ["#e3f2fd", "#fff3e0", "#fff3e0", "#ede7f6", "#c8e6c9"]
for i, (s, c) in enumerate(zip(stages, colours)):
    x = 0.3 + i * 2.15
    ax.add_patch(Rectangle((x, 0.5), 1.75, 1.15, facecolor=c,
                           edgecolor="#37474f", linewidth=1.5))
    ax.text(x + 0.87, 1.07, s, ha="center", va="center", fontsize=7.5)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + 2.08, 1.07), (x + 2.25, 1.07),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#78909c"))
ax.text(5.5, 0.15, "Pipeline(stages=[...]).fit(train)  →  one PipelineModel",
        ha="center", fontsize=9, color="#37474f", fontweight="bold")
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 2.0)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
**`VectorAssembler` is the one that has no scikit-learn equivalent, and everyone
meets it in their first hour of MLlib.** Every MLlib model expects its input as a
**single column** containing a vector — not many separate columns. `VectorAssembler`
is the Transformer that bundles your feature columns into that one `features`
column. If you forget it, you get a type error that makes no sense until you know
this.

Here is what it actually produces:
"""
)
st.code(res["features_show"], language="text")

st.header("3 · Results — and the lesson you've now seen three times")
results = pd.read_csv(io.StringIO(res["results_csv"]))

fig, ax = plt.subplots(figsize=(9, 3.0))
x = np.arange(2)
w = 0.35
for i, feat in enumerate(["raw columns", "engineered in Spark"]):
    sub = results[results.features == feat]
    ax.bar(x + (i - 0.5) * w, sub["auc"], w, label=feat,
           color=["#90a4ae", "#2e7d32"][i], edgecolor="white")
    for j, v in enumerate(sub["auc"]):
        ax.text(j + (i - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center",
                fontsize=8.5)
ax.axhline(0.5, color="#c62828", linestyle="--", linewidth=1.2)
ax.text(1.35, 0.505, "random guessing", fontsize=7.5, color="#c62828")
ax.set_xticks(x)
ax.set_xticklabels(["LogisticRegression", "RandomForest"])
ax.set_ylabel("AUC")
ax.set_ylim(0.45, 0.65)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
st.pyplot(fig)
plt.close(fig)
st.dataframe(
    results.style.format({"auc": "{:.4f}", "fit_seconds": "{:.1f}"})
    .highlight_max(subset=["auc"], color="#c8e6c9"),
    hide_index=True, width="stretch")
st.caption(f"📦 {artifacts.provenance(res)} — four real MLlib pipelines on "
           f"{int(res['n_train']):,} rows.")

raw_best = results[results.features == "raw columns"]["auc"].max()
eng_best = results[results.features == "engineered in Spark"]["auc"].max()
st.success(
    f"""
**Raw columns: AUC {raw_best:.3f}. Features engineered in Spark: AUC
{eng_best:.3f}.** A gain of **+{eng_best - raw_best:.3f}** — and the *model* never
changed.

The engineered features were computed **in Spark itself**, with the verbs from
two pages ago:

- **`striker_rate`** — each batter's historical boundary rate, from
  `groupBy("striker").agg(avg("label"))` on the **training split only**, joined
  back on. (This is *target encoding*; computing it on the full data would leak
  the answer — the Section 2 rule, still true at scale.)
- **`bowler_rate`** — the same for bowlers. Some bowlers are simply harder to hit.
- **`is_death`, `is_powerplay`** — the phase of the innings, which the chart above
  showed is the dominant pattern.

Notice also that **RandomForest is *worse* than LogisticRegression here**, on both
feature sets. The fancier model lost. If there were one sentence to carry out of
this entire course, it is the one you have now met in Section 3, Section 7 and
again here:

> **Better features beat better models, almost every time.**
"""
)

st.header("4 · The code")
show_example(
    experiments.SNIPPETS["s7_pipeline"],
    """
- `train.cache()` — **the single most important performance line in most MLlib scripts.** Remember from the lazy-evaluation page that Spark *recomputes* a DataFrame every time you use it. Training touches `train` dozens of times, so without `cache()` you'd re-read and re-filter the parquet on every pass. `cache()` says "keep this in memory". Forgetting it is the most common reason a beginner's MLlib job takes 20 minutes instead of 20 seconds.
- The `striker_rate` block — feature engineering with the plain DataFrame verbs. Note it's computed from `train` **only**, then joined onto both splits. Leakage discipline doesn't change just because the data got big.
- `.fillna({"striker_rate": base})` — a batter in the test set who never appeared in training gets the global average. **Always handle unseen categories**; in production they are guaranteed to appear.
- `Pipeline(stages=[...])` — the stages run in order. `VectorAssembler` bundles the columns; `StandardScaler` learns μ and σ **from the training data**; `LogisticRegression` learns the weights.
- `pipeline.fit(train)` → a **`PipelineModel`**. This one object holds the fitted scaler *and* the fitted classifier. `model.transform(test)` replays every step with the *training* statistics — which is exactly why the pipeline abstraction prevents the leakage bug of re-fitting the scaler on your test data.
- `model.write().overwrite().save("path")` (not shown) — save the whole fitted pipeline to disk and load it in a production job. That is how a model actually ships.
""",
    key="s7_pipeline",
    heavy=True,
    est="~30 s",
)

guided_sandbox(
    key="s7",
    heavy=True,
    defer=True,
    est="~3 min",
    steps="""
1. **Step 1** — build the label (`runs_off_bat >= 4`), split 80/20 with
   `randomSplit([0.8, 0.2], seed=42)`, and **`.cache()` the training set**.
   Print the base rate.
2. **Step 2** — build a pipeline with just `VectorAssembler` + `StandardScaler`
   + `LogisticRegression` on the raw columns `["over", "innings"]`. Fit it and
   print the AUC. It should be poor — that's the baseline.
3. **Step 3** — engineer `bowler_rate` in Spark (group the **training** data by
   bowler, average the label, filter to bowlers with ≥200 balls, join it back).
   Add it to the pipeline and re-measure the AUC. How much did it move?
4. **Step 4 (stretch)** — swap `LogisticRegression` for
   `RandomForestClassifier(numTrees=20, maxDepth=6)` and compare. Then try
   `GBTClassifier`. Does any model beat the gain you got from *one* good
   feature?
""",
    setup_code='''from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F
from utils.spark import build_session, load_deliveries

spark = build_session()
df = (load_deliveries(spark)
      .filter(F.col("is_wide") == 0)
      .filter(F.col("year") >= 2015))
print("rows:", df.count())

# Step 1: label, split, cache, base rate


# Step 2: baseline pipeline on raw columns -> AUC


# Step 3: engineer bowler_rate in Spark, add it -> AUC


# Step 4 (stretch): try RandomForest / GBT - can a model beat a good feature?


spark.stop()
''',
    solution_code='''from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
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
train.cache()
base = train.select(F.avg("label")).first()[0]
print(f"base rate: {base:.3%} of balls go to the boundary")

ev = BinaryClassificationEvaluator(metricName="areaUnderROC")

def run(cols, tr, te, clf=None):
    clf = clf or LogisticRegression(maxIter=20)
    pipe = Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features"),
        clf])
    model = pipe.fit(tr)
    return ev.evaluate(model.transform(te))

auc0 = run(["over", "innings"], train, test)
print(f"\\nbaseline (over + innings):        AUC {auc0:.4f}")

bowler_rate = (train.groupBy("bowler")
               .agg(F.avg("label").alias("bowler_rate"),
                    F.count("*").alias("n"))
               .filter(F.col("n") >= 200)
               .select("bowler", "bowler_rate"))

def enrich(d):
    return (d.join(bowler_rate, "bowler", "left")
             .fillna({"bowler_rate": base})
             .withColumn("is_death", (F.col("over") >= 40).cast("double")))

tr_e, te_e = enrich(train), enrich(test)
cols = ["over", "innings", "bowler_rate", "is_death"]
auc1 = run(cols, tr_e, te_e)
print(f"+ bowler_rate + is_death:         AUC {auc1:.4f}  ({auc1-auc0:+.4f})")

auc_rf = run(cols, tr_e, te_e,
             RandomForestClassifier(numTrees=20, maxDepth=6, seed=42))
print(f"same features, RandomForest:      AUC {auc_rf:.4f}  "
      f"({auc_rf-auc1:+.4f} vs LogisticRegression)")
print("\\nONE good feature beat swapping the whole model. Every time.")

spark.stop()''',
)
