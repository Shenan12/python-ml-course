"""Getting a SparkSession, safely, on a laptop.

Two things here matter beyond this course:

1. **The Windows Python-worker fix.** Spark launches Python "executor" processes
   by running the command `python`. On Windows that often hits the Microsoft
   Store alias stub and you get the baffling error
   `Python worker failed to connect back`. The fix is to tell Spark exactly
   which interpreter to use, via `PYSPARK_PYTHON`. This bites almost everyone
   once; now it won't bite you.

2. **A SparkSession is expensive** (~10-20 s to start: it boots a JVM). You
   create ONE and reuse it. In Streamlit that means `@st.cache_resource`, which
   caches the object itself rather than its return value.

We run in **local mode** (`local[*]`) throughout: Spark runs entirely inside one
process on your laptop, using your CPU cores as if they were a cluster's
machines. The API is *identical* to the one you'd use on a 500-node cluster —
which is the whole point. Learn it here, deploy it there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERIES = ROOT / "data" / "odi_deliveries.parquet"
INNINGS = ROOT / "data" / "odi_batting_innings.csv"
JDK_DIR = ROOT / ".jdk"


def _java_major(java_exe: str) -> int | None:
    """Parse the major version out of `java -version` (e.g. 17, 21, 25)."""
    import re
    import subprocess
    try:
        out = subprocess.run([java_exe, "-version"], capture_output=True,
                             text=True, timeout=30)
        m = re.search(r'version "(\d+)', out.stderr + out.stdout)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def find_java_home() -> str | None:
    """Find a JDK that Spark can actually use.

    Spark 4 supports **Java 17 and 21**. It does NOT support Java 22+: those
    versions removed the Security Manager APIs that Hadoop's file-access layer
    still calls, and you get the deeply unhelpful
    `UnsupportedOperationException: getSubject is not supported` the moment you
    read a file.

    Search order:
      1. a private JDK bundled in `.jdk/` (how this course runs on a Windows
         machine whose system Java is 25 — the system Java is untouched);
      2. the system Java, IF it is version 17 or 21 (this is how a deployed
         Streamlit Cloud app finds the JDK installed via `packages.txt`).
    """
    if JDK_DIR.is_dir():
        for candidate in sorted(JDK_DIR.glob("jdk-1[78]*")) + \
                sorted(JDK_DIR.glob("jdk-21*")):
            if (candidate / "bin" / "java.exe").exists() or \
                    (candidate / "bin" / "java").exists():
                return str(candidate)
    # fall back to the system Java if it's a supported version
    import shutil
    system_java = shutil.which("java")
    if system_java:
        major = _java_major(system_java)
        if major in (17, 21):
            return None          # None = "leave JAVA_HOME alone, system is fine"
    return "UNSUPPORTED"


def configure_env() -> None:
    """Prepare the environment BEFORE Spark starts. Two fixes live here."""
    # 1. Spark launches its Python workers by running `python`. On Windows that
    #    hits the Microsoft Store alias stub and fails with the baffling
    #    "Python worker failed to connect back". Point it at THIS interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # 2. Use a Spark-supported JDK (17/21).
    java_home = find_java_home()
    if java_home == "UNSUPPORTED":
        raise RuntimeError(
            "Spark 4 needs Java 17 or 21, and none was found. "
            "Locally: install Temurin 17 (https://adoptium.net) or unzip one "
            "into the project's .jdk/ folder. "
            "On Streamlit Community Cloud: add a `packages.txt` file "
            "containing the line `openjdk-17-jre-headless` (this repo ships "
            "one). Note Java 22+ does NOT work — Spark's file layer fails "
            "with 'getSubject is not supported'."
        )
    if java_home:
        os.environ["JAVA_HOME"] = java_home


def build_session(app_name: str = "cricket-spark", shuffle_partitions: int = 8):
    """A small, quiet, laptop-friendly SparkSession in local mode."""
    import tempfile

    configure_env()
    from pyspark.sql import SparkSession

    # Spark scribbles scratch files as it works: shuffle spills, the SQL
    # warehouse directory, Derby's metastore log. By default those land in the
    # CURRENT WORKING DIRECTORY — which on a deployed app may be read-only and
    # give you PermissionError. Point every one of them at the temp dir, which
    # is writable everywhere.
    scratch = Path(tempfile.gettempdir()) / "ml_course_spark"
    scratch.mkdir(exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                    # every core on THIS machine
        .config("spark.ui.enabled", "false")   # no web UI: we're embedded
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "false")  # keep plans predictable
        .config("spark.local.dir", str(scratch / "local"))
        .config("spark.sql.warehouse.dir", (scratch / "warehouse").as_uri())
        .config("spark.driver.extraJavaOptions",
                f"-Dderby.system.home={scratch / 'derby'}")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_session():
    """Streamlit-cached session: built once, reused for the whole app run."""
    import streamlit as st

    @st.cache_resource(show_spinner="Starting a SparkSession (this boots a "
                                    "JVM — ~15 s, once)…")
    def _session():
        return build_session()

    return _session()


def load_deliveries(spark):
    """The ball-by-ball table: ~1.35 million rows."""
    return spark.read.parquet(str(DELIVERIES))
