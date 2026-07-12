"""A tiny disk-backed store for precomputed results.

Why this exists
---------------
Some pages in Section 6 (and later, Section 7) train real neural networks.
Training them on every page load would make the app painfully slow on a
laptop. So instead we run the training ONCE, in `build_artifacts.py`, and save
the results here. The pages then load them instantly.

This does NOT mean the numbers are made up. Every value in `artifacts/` was
produced by actually executing the code — just at build time rather than at
page-render time. Three things keep that promise honest:

1. Every artifact carries **provenance**: when it was computed, with which
   seed, and with which library versions.
2. Code snippets are stored with a **hash of the exact code that produced
   them**. If the code on the page ever changes without the artifact being
   rebuilt, the page shows a warning instead of a stale number.
3. Every page has a **▶ Run live** button that recomputes the result on your
   machine, so you can always check the shipped numbers yourself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _versions() -> dict:
    versions = {}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        pass
    versions["numpy"] = np.__version__
    return versions


def code_hash(code: str) -> str:
    """A fingerprint of a code snippet, used to detect drift between the code
    shown on a page and the output stored for it."""
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:16]


def _paths(name: str) -> tuple[Path, Path]:
    return ART_DIR / f"{name}.json", ART_DIR / f"{name}.npz"


def save(name: str, payload: dict, seed: int | None = None) -> None:
    """Save a result. NumPy arrays go to a .npz, everything else to a .json."""
    ART_DIR.mkdir(exist_ok=True)
    json_path, npz_path = _paths(name)

    arrays = {k: np.asarray(v) for k, v in payload.items()
              if isinstance(v, np.ndarray)}
    plain = {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}

    plain["_meta"] = {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "seed": seed,
        "versions": _versions(),
        "array_keys": sorted(arrays),
    }
    json_path.write_text(json.dumps(plain, indent=2, default=str),
                         encoding="utf-8")
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    elif npz_path.exists():
        npz_path.unlink()


def load(name: str) -> dict | None:
    """Load a result, or return None if it hasn't been built yet.

    Never raises: a missing or corrupt artifact simply means "not available",
    and the calling page falls back to a ▶ Run live button.
    """
    json_path, npz_path = _paths(name)
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as npz:
                payload.update({k: npz[k] for k in npz.files})
        return payload
    except Exception:  # noqa: BLE001 - a broken artifact is just a cache miss
        return None


def exists(name: str) -> bool:
    return _paths(name)[0].exists()


def provenance(payload: dict | None) -> str:
    """A one-line 'where did this number come from' caption."""
    if not payload or "_meta" not in payload:
        return "computed live just now"
    meta = payload["_meta"]
    bits = [f"precomputed {meta.get('computed_at', '?')}"]
    if meta.get("seed") is not None:
        bits.append(f"seed {meta['seed']}")
    versions = meta.get("versions", {})
    if "torch" in versions:
        bits.append(f"torch {versions['torch']}")
    return " · ".join(bits)
