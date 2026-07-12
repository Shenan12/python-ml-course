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


def find_java_home() -> str | None:
    """Find a JDK that Spark can actually use.

    Spark 4 supports **Java 17 and 21**. It does NOT support Java 22+: those
    versions removed the Security Manager APIs that Hadoop's file-access layer
    still calls, and you get the deeply unhelpful
    `UnsupportedOperationException: getSubject is not supported` the moment you
    read a file.

    This machine's system Java is 25, so the course ships a private JDK 17 in
    `.jdk/` and points Spark at it — without touching your system Java.
    """
    if JDK_DIR.is_dir():
        for candidate in sorted(JDK_DIR.glob("jdk-1[78]*")) + \
                sorted(JDK_DIR.glob("jdk-21*")):
            if (candidate / "bin" / "java.exe").exists() or \
                    (candidate / "bin" / "java").exists():
                return str(candidate)
    return None


def configure_env() -> None:
    """Prepare the environment BEFORE Spark starts. Two fixes live here."""
    # 1. Spark launches its Python workers by running `python`. On Windows that
    #    hits the Microsoft Store alias stub and fails with the baffling
    #    "Python worker failed to connect back". Point it at THIS interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # 2. Use a Spark-supported JDK (17/21), not the system's Java 25.
    java_home = find_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home


def build_session(app_name: str = "cricket-spark", shuffle_partitions: int = 8):
    """A small, quiet, laptop-friendly SparkSession in local mode."""
    configure_env()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                    # every core on THIS machine
        .config("spark.ui.enabled", "false")   # no web UI: we're embedded
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "false")  # keep plans predictable
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
