# Python → Machine Learning → Survival Analysis: an interactive course

A local, interactive Streamlit application that teaches Python and machine
learning from scratch, ending with full, paper-grounded coverage of
**DeepSurv** (Katzman et al. 2018) and **DeepHit** (Lee et al. 2018) — the
two papers behind the author's dissertation.

Every topic page has three parts:

1. a plain-English explanation written for a true beginner,
2. a visualization of the *mechanism* (not just a results chart),
3. a 🧪 **"try it yourself"** code editor you can change and re-run inside
   the app — no terminal needed.

All outputs shown on the pages are computed live on your machine when the
page loads; nothing is pasted in.

---

## Setup (beginner-friendly, step by step)

### 0. What you need first

- **Python 3.11 or newer** — check by opening a terminal (on Windows: press
  the Windows key, type `powershell`, press Enter) and running:

  ```
  python --version
  ```

  If that prints an error, install Python from https://www.python.org/downloads/
  and **tick "Add python.exe to PATH"** during the installer.

### Before you start: Java for PySpark ☕ (the one non-pip dependency)

> **⚠️ Already handled on this machine — read this anyway, because it's the
> gotcha that catches everyone.**
>
> **Spark 4 supports Java 17 and 21 — and NOT Java 22+.** Newer Java versions
> removed the Security Manager APIs that Spark's file-reading layer still calls,
> so the moment you read a file you get the deeply unhelpful error:
>
> ```
> UnsupportedOperationException: getSubject is not supported
> ```
>
> This machine's system Java is **25**, so the course ships its own **JDK 17** in
> the `.jdk/` folder and points Spark at it (see `utils/spark.py`). Your system
> Java is untouched. If you ever move this project to another machine, either
> install a JDK 17/21 or re-download one into `.jdk/`.
>
> **The second Windows gotcha:** Spark starts its Python workers by running the
> command `python`, which on Windows can hit the Microsoft Store alias stub and
> fail with `Python worker failed to connect back`. `utils/spark.py` fixes this
> by setting `PYSPARK_PYTHON` to the exact interpreter you're running.

Section 4 of the course uses **PySpark**, and Spark itself runs on **Java**.
`pip` cannot install Java for you — it's a separate program.

1. Check whether you already have it. In a terminal, run:

   ```
   java -version
   ```

   If you see a version number of **17 or higher** (e.g. `openjdk version
   "17.0.12"`), you are done — skip to step 1 below.

2. If not, install a free OpenJDK build, e.g. **Eclipse Temurin 17 (LTS)**
   from https://adoptium.net/ . On Windows, during installation enable the
   options to **set JAVA_HOME** and **add to PATH**.

3. Close and reopen your terminal, then run `java -version` again to
   confirm it now works.

> You can do everything in Sections 1–3 and 5–7 without Java — only the
> PySpark section needs it.

### 1. Get the code and install the Python packages

In a terminal, from the folder that contains this README:

```
pip install -r requirements.txt
```

This installs Streamlit, NumPy, pandas, matplotlib, plotly, scikit-learn,
PySpark, PyTorch, and the survival-analysis libraries (pycox, torchtuples,
lifelines). The PyTorch download is large (roughly a few hundred MB), so
this step can take several minutes.

### 2. Run the app

Still in the same folder:

```
python -m streamlit run app.py
```

(The plain `streamlit run app.py` also works if the `streamlit` command is
on your PATH; the `python -m` form works either way.)

Your browser opens at `http://localhost:8501`. Use the **sidebar** to
navigate. Start at *1 · Python Fundamentals* and work downward — each page
assumes only the pages before it.

To stop the app: go back to the terminal and press `Ctrl+C`.

### 3. Precomputed results (why the app starts instantly)

Section 6 (DeepSurv) trains real neural networks. Training them every time you
opened a page would make the app crawl on a laptop, so the results are
**precomputed once** and stored in `artifacts/`. The app ships with them, so it
starts instantly.

This does **not** mean the numbers are made up. Every value was produced by
actually running the code — just ahead of time instead of while you wait. Three
things keep that honest:

- each result carries a visible **timestamp, seed and library versions**;
- each code snippet is stored with a **hash of the exact code that produced
  it**, so if the code ever changes without a rebuild, the page hides the stale
  number and says so;
- **every heavy demo has a `▶ Run live` button** — press it and the training
  runs on your machine, right there, and the numbers refresh.

To regenerate everything yourself:

```
python build_artifacts.py
```

That takes a few minutes and rewrites `artifacts/`. You only need it if you
edit one of the heavy snippets (the app will tell you if you have).

### 4. Using the code sandboxes

Every page ends with a 🧪 **Try it yourself** editor:

- Click into the code, edit anything.
- Press **Ctrl+Enter**, or click the small **APPLY** button at the editor's
  bottom-right corner, to run your version.
- Errors are caught and explained under the editor — you cannot break the
  app. Refreshing the page restores the starter code.

---

## Course status

| # | Section | Status |
|---|---------|--------|
| 1 | Python Fundamentals | ✅ built & verified |
| 2 | ML Foundations | ✅ built & verified |
| 3 | Classical ML Algorithms | ✅ built & verified |
| 4 | PySpark & Big Data | 🚧 not built yet (deferred by request) |
| 5 | Neural Networks | ✅ built & verified |
| 6 | DeepSurv, in full | ✅ built & verified |
| 7 | DeepHit, in full | ✅ built & verified |

## The cricket dataset (Section 7)

Section 7 uses **real men's ODI ball-by-ball data** from
[Cricsheet](https://cricsheet.org) (CC BY 4.0), turned into a survival dataset:

- one row per **batter-innings** (25,503 of them, 2010–2026)
- **duration** = balls faced · **event** = dismissed · **censored** = NOT OUT

A not-out batter is genuinely right-censored — the innings ended while he was
still batting — so this is real censoring, not simulated.

The prepared file ships with the course (`data/odi_batting_innings.csv`). To
rebuild it from the original Cricsheet archive:

```
python scripts/prepare_cricket_data.py
```

## Papers covered in depth (Sections 6–7)

- Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., &
  Kluger, Y. (2018). *DeepSurv: personalized treatment recommender system
  using a Cox proportional hazards deep neural network.* BMC Medical
  Research Methodology, 18(1), 24.
- Lee, C., Zame, W., Yoon, J., & van der Schaar, M. (2018). *DeepHit: A
  Deep Learning Approach to Survival Analysis with Competing Risks.*
  Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).

## Deploying to Streamlit Community Cloud

The app is deployable, and several things in this repo exist specifically to
make that work. If you fork/redeploy, keep them:

1. **Commit `data/` and `artifacts/`.** The app reads both at runtime
   (~7 MB + ~3 MB). Without them, pages show a "run build_artifacts.py" notice
   instead of results.
2. **`packages.txt`** installs **Java 17** on the cloud machine — required by
   the PySpark section's ▶ Run buttons. (Spark 4 needs Java 17/21; Java 22+
   fails with `getSubject is not supported`.)
3. **`requirements.txt` starts with a PyTorch CPU index line.** Without it,
   Linux pip installs the multi-gigabyte CUDA build of torch and the deploy
   blows its resource limits.
4. **`.jdk/` is gitignored** (it's a 304 MB Windows convenience for this
   machine). The cloud gets Java from `packages.txt` instead.
5. **All scratch files go to the system temp folder** — the File I/O page's
   demo files and Spark's shuffle/warehouse/Derby files. A deployed app's
   project folder may be read-only; writing there is what causes
   `PermissionError` crashes. If you add new pages, follow the same rule:
   never write into the repo folder at runtime.

Heads-up on expectations: the heavy pages show **precomputed results**
everywhere, so the deployed app is fast. The ▶ "Run live" buttons genuinely
retrain on the cloud machine — the DeepSurv/DeepHit ones take their stated
times; the Spark ones additionally boot a JVM (~20 s extra on first press).

## Project layout

```
app.py                       entry point — streamlit run app.py
requirements.txt             python dependencies (note the torch CPU index line)
packages.txt                 apt packages for Streamlit Cloud (Java 17 for Spark)
build_artifacts.py           re-runs every heavy computation and refreshes artifacts/
utils/
  sandbox.py                 the "run this code and show the result" engine
  artifacts.py               the precomputed-results store (provenance + code hashes)
  mockdata.py                shared mock datasets (patients, pumps, machines)
  cricket.py                 the ODI survival dataset loader + feature engineering
  spark.py                   SparkSession builder (Java detection, temp dirs, fixes)
scripts/
  prepare_cricket_data.py    rebuilds data/odi_batting_innings.csv from Cricsheet
  prepare_spark_data.py      rebuilds data/odi_deliveries.parquet from Cricsheet
sections/
  home.py                    welcome page
  section1_python/           p1–p8   Python fundamentals
  section2_ml_foundations/   m1–m7   ML foundations
  section3_classical_ml/     c1–c10  classical algorithms (incl. AIC)
  section4_pyspark/          s1–s8   PySpark & big data
  section5_neural_networks/  n1–n7   neural networks
  section6_deepsurv/         d1–d8   DeepSurv, in full
  section7_deephit/          h1–h7   DeepHit, in full
data/                        the real cricket datasets (committed — the app needs them)
artifacts/                   precomputed results (committed — the app needs them)
.jdk/                        local-only Java 17 for Spark (gitignored)
```
