"""Small runtime fixes so the course runs on read-only deployments.

On Streamlit Community Cloud the app's folder and the installed packages are
**read-only** — only the temp directory is writable. Two of our libraries
assume they can write wherever they like, which crashes every page that
imports them. This module redirects both to the temp directory.

Importing this module (done at the top of ``app.py``) applies the cheap fixes:

1. ``import pycox`` creates a dataset-download folder *inside its own
   site-packages install* at import time. pycox honours the
   ``PYCOX_DATA_DIR`` environment variable, so we point it at temp.

2. pycox still calls ``scipy.integrate.simps``, which SciPy renamed to
   ``simpson``. Alias it back.

``patch_torchtuples()`` (called by the ``_experiments`` modules, which import
torch anyway) applies the third:

3. ``torchtuples.callbacks.EarlyStopping`` checkpoints the best weights to a
   file named by ``make_name_hash()`` — a bare relative filename, i.e. the
   current working directory, which on a deployment is the read-only repo
   mount. We wrap ``make_name_hash`` so those checkpoints land in temp too.

None of this changes any result — it only changes *where scratch files go*.
It is also exactly the kind of fix you will write yourself one day: code that
works on your laptop, deployed somewhere it isn't allowed to write.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SCRATCH = Path(tempfile.gettempdir()) / "ml_course_scratch"
SCRATCH.mkdir(exist_ok=True)

# --- 1. pycox: download/cache dir out of site-packages -----------------------
os.environ.setdefault("PYCOX_DATA_DIR", str(SCRATCH / "pycox_data"))
Path(os.environ["PYCOX_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

# --- 2. scipy: pycox expects the pre-1.14 name -------------------------------
import scipy.integrate as _si  # noqa: E402

if not hasattr(_si, "simps"):
    _si.simps = _si.simpson


# --- 3. torchtuples: EarlyStopping checkpoints go to temp --------------------
def patch_torchtuples() -> None:
    """Make torchtuples write its weight-checkpoint files to temp.

    Must run before any ``EarlyStopping(...)`` is constructed. Kept out of
    the module level so that importing ``compat`` stays cheap — this is the
    only part that pulls in torch.
    """
    import torchtuples.callbacks as ttcb
    import torchtuples.utils as ttu

    if getattr(ttu.make_name_hash, "__module__", "") == __name__:
        return  # already patched

    original = ttu.make_name_hash

    def make_name_hash_in_temp(name="", file_ending=".pt"):
        return str(SCRATCH / original(name, file_ending))

    make_name_hash_in_temp.__module__ = __name__
    ttu.make_name_hash = make_name_hash_in_temp
    # callbacks.py did `from torchtuples.utils import make_name_hash`, so it
    # holds its own reference — rebind that one too.
    ttcb.make_name_hash = make_name_hash_in_temp
