"""Shared helpers used by every page of the course.

Two building blocks:

- show_example(code, explanation): displays a code snippet, an optional
  line-by-line explanation, then ACTUALLY EXECUTES the snippet on your
  machine and shows whatever it printed or plotted. Nothing you see in an
  "Output" box is hard-coded — it is computed live every time the page loads.

- sandbox(starter_code, key): an editable code editor (streamlit-ace).
  Whatever code is in the editor gets executed in a controlled namespace,
  and the printed output / plots are shown below it. Errors are caught and
  shown as a friendly message instead of crashing the page.
"""

import contextlib
import io
import math
import random
import traceback

import matplotlib
matplotlib.use("Agg")  # draw to memory, never try to open a desktop window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def _execute(code: str):
    """Run `code` in a fresh namespace.

    Returns (stdout_text, list_of_matplotlib_figures, error_info or None).
    error_info is a tuple (short_message, full_traceback_text).
    """
    namespace = {
        "__name__": "__sandbox__",
        "np": np,
        "pd": pd,
        "plt": plt,
        "math": math,
        "random": random,
    }
    buffer = io.StringIO()
    error_info = None

    figures_before = set(plt.get_fignums())
    try:
        compiled = compile(code, "<your code>", "exec")
        with contextlib.redirect_stdout(buffer):
            exec(compiled, namespace)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: user code can raise anything
        lineno = None
        for frame in traceback.extract_tb(exc.__traceback__):
            if frame.filename == "<your code>":
                lineno = frame.lineno
        if isinstance(exc, SyntaxError) and exc.filename == "<your code>":
            lineno = exc.lineno
        where = f" (line {lineno} of your code)" if lineno else ""
        short = f"{type(exc).__name__}: {exc}{where}"
        error_info = (short, traceback.format_exc())

    new_figs = [plt.figure(num) for num in plt.get_fignums()
                if num not in figures_before]
    return buffer.getvalue(), new_figs, error_info


def _render_result(stdout_text, figures, error_info, output_label):
    if error_info is not None:
        short, full_tb = error_info
        st.error(
            f"**Your code raised an error — that is completely normal when "
            f"experimenting.**\n\n`{short}`\n\nFix the line mentioned above "
            f"and run again."
        )
        with st.expander("Show the full error report (traceback)"):
            st.code(full_tb, language="text")
    if stdout_text:
        st.caption(output_label)
        st.code(stdout_text, language="text")
    for fig in figures:
        st.pyplot(fig)
        plt.close(fig)
    if not stdout_text and not figures and error_info is None:
        st.caption(
            "The code ran without errors but printed nothing. "
            "Add a `print(...)` to see values."
        )


def _heavy_output(code: str, key: str, run_label: str, output_label: str,
                  est: str = "~20 s"):
    """Show the saved output of an expensive snippet, plus a button to re-run
    it live.

    The snippet is NOT executed on page load — these train real neural networks
    and would make every click cost tens of seconds. Instead we show the output
    that `build_artifacts.py` captured by running this exact code, and let you
    recompute it on demand.

    Honesty guard: the artifact stores a hash of the code that produced it. If
    the code on the page has changed since the last build, we refuse to show
    the stale output and say so.
    """
    from utils import artifacts

    payload = artifacts.load(f"snippet_{key}")
    ran_key = f"_ran_{key}"

    if st.button(f"▶ Run this code now ({est})", key=f"_btn_{key}"):
        st.session_state[ran_key] = True

    if st.session_state.get(ran_key):
        with st.spinner("Training for real — this is the actual computation…"):
            stdout_text, figures, error_info = _execute(code)
        _render_result(stdout_text, figures, error_info,
                       "Output — just computed live on your machine:")
        return

    if payload is None:
        st.info(
            "No saved output for this snippet yet. Press the button above to "
            "run it, or run `python build_artifacts.py` once to precompute "
            "every heavy result in the course."
        )
        return

    if payload.get("code_hash") != artifacts.code_hash(code):
        st.warning(
            "⚠️ **This snippet has been edited since its output was last "
            "computed**, so the saved output no longer belongs to the code "
            "above. Rather than show you a stale number, we're hiding it — "
            "press **▶ Run this code now** (or rerun `build_artifacts.py`)."
        )
        return

    st.caption(f"{output_label} — {artifacts.provenance(payload)}. "
               "Nothing here is hand-typed: it is the captured output of "
               "running the exact code above. Press ▶ to recompute it live.")
    st.code(payload.get("stdout", ""), language="text")


def show_example(code: str, explanation: str | None = None,
                 key: str | None = None, heavy: bool = False,
                 est: str = "~20 s"):
    """Show a snippet, its line-by-line explanation, and its output.

    By default the snippet is executed live on every page load, so what you see
    is genuinely what the code does. Snippets that train neural networks pass
    `heavy=True` (with a `key`): those show their saved output and offer a
    button to run live, so the page stays fast.
    """
    st.code(code, language="python")
    if explanation:
        with st.expander("📖 What each line does (click to open)"):
            st.markdown(explanation)
    if heavy:
        if not key:
            raise ValueError("show_example(heavy=True) requires a key")
        _heavy_output(code, key, "▶ Run this code now",
                      "Output of the code above", est=est)
        return
    stdout_text, figures, error_info = _execute(code)
    _render_result(
        stdout_text, figures, error_info,
        "Output — computed live on your machine just now, not pasted in:",
    )


def guided_sandbox(key: str, steps: str, setup_code: str, solution_code: str,
                   height: int | None = None, heavy: bool = False,
                   est: str = "~20 s", defer: bool = False):
    """A do-it-yourself sandbox: numbered instructions + an editor that
    contains only the data setup and '# Step N' prompts. A worked solution
    sits in a collapsed expander.

    For ordinary pages the solution is executed live, so its output is always
    verified. For `heavy=True` pages (which train real networks) the solution
    shows its saved output plus a ▶ Run button — otherwise merely *opening* the
    page would cost tens of seconds, because Streamlit executes an expander's
    body whether or not it is open."""
    st.divider()
    st.subheader("🧪 Try it yourself — you write the code this time")
    st.markdown(
        "The editor below only sets up the data. Follow the numbered steps, "
        "typing your code under each `# Step` comment, then press "
        "**Ctrl+Enter** (or the **APPLY** button at the editor's "
        "bottom-right) to run. Errors are safe — read them, fix, re-run."
    )
    st.markdown(steps)
    if height is None:
        height = max(220, min(560, 19 * (setup_code.count("\n") + 2)))

    code = None
    try:
        from streamlit_ace import st_ace
        code = st_ace(
            value=setup_code,
            language="python",
            theme="tomorrow_night",
            key=f"ace_{key}",
            height=height,
            font_size=14,
            wrap=True,
            auto_update=False,
        )
    except Exception:
        code = st.text_area("Code", value=setup_code, height=height,
                            key=f"ta_{key}")
    if not code:
        code = setup_code

    if defer:
        # Some sandboxes (every Spark one) cost real time just to START —
        # a SparkSession boots a JVM. Running the editor's contents on every
        # page render would make the page unusable, so we wait for a click.
        run_key = f"_run_{key}"
        if st.button(f"▶ Run my code ({est})", key=f"_runbtn_{key}",
                     type="primary"):
            st.session_state[run_key] = True
        if st.session_state.get(run_key):
            with st.spinner("Running your code…"):
                stdout_text, figures, error_info = _execute(code)
            _render_result(stdout_text, figures, error_info,
                           "Output of YOUR code in the editor above:")
        else:
            st.caption(
                "Press **▶ Run my code** when you're ready. (This one starts a "
                "SparkSession, which boots a JVM — so it doesn't run "
                "automatically.)"
            )
    else:
        stdout_text, figures, error_info = _execute(code)
        _render_result(stdout_text, figures, error_info,
                       "Output of YOUR code in the editor above:")

    with st.expander("🔑 Stuck? Reveal a worked solution"):
        st.markdown("Try each step yourself before peeking — struggling "
                    "first is what makes it stick.")
        st.code(solution_code, language="python")
        if heavy:
            _heavy_output(solution_code, f"{key}_solution",
                          "▶ Run the solution now",
                          "The solution's output", est=est)
        else:
            sol_out, sol_figs, sol_err = _execute(solution_code)
            _render_result(sol_out, sol_figs, sol_err,
                           "The solution's output — computed live just now:")


def sandbox(starter_code: str, key: str, height: int | None = None,
            intro: str | None = None):
    """An editable, re-runnable code box, pre-filled with working code."""
    st.divider()
    st.subheader("🧪 Try it yourself")
    st.markdown(
        intro
        or "Edit the code below, then press **Ctrl+Enter** (or click the "
           "**APPLY** button at the bottom-right of the editor) to re-run it. "
           "Break things on purpose — errors here are safe and the starter "
           "code is always one page-refresh away."
    )
    if height is None:
        height = max(180, min(560, 19 * (starter_code.count("\n") + 2)))

    code = None
    try:
        from streamlit_ace import st_ace
        code = st_ace(
            value=starter_code,
            language="python",
            theme="tomorrow_night",
            key=f"ace_{key}",
            height=height,
            font_size=14,
            wrap=True,
            auto_update=False,  # False => an APPLY button appears in the editor
        )
    except Exception:  # streamlit-ace missing or broken -> plain text box
        code = st.text_area("Code", value=starter_code, height=height,
                            key=f"ta_{key}")

    if not code:
        code = starter_code

    stdout_text, figures, error_info = _execute(code)
    _render_result(stdout_text, figures, error_info,
                   "Output of the code in the editor above:")
