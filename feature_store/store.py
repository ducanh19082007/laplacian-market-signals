"""
feature_store.store -- append-only, timestamped log of the L4 regime pipeline.

INDEPENDENT of L1-L4: it receives plain dicts and writes them to a JSONL file (one
JSON object per line) under data/ (gitignored). One file PER RUN, so a crashed
session never corrupts earlier data; the offline step globs data/*.jsonl and
concatenates. This is the data-gathering half of improvement B -- collection here,
model training/analysis entirely offline later (see label.py).

Why JSONL:
  * append-only + crash-safe -- a torn final line is trivially skipped on read,
    and every line is flushed, so Ctrl-C keeps everything written so far;
  * schema-flexible -- add a field later and old rows simply lack it;
  * loads straight into pandas: pd.read_json(path, lines=True).
Convert to Parquet in the offline labelling step if a model wants columnar.

Forward labels are NOT written here -- they can't be known at write time (a row's
label is what happens H seconds LATER). Every row carries an absolute wall-clock
`ts`, so label.py can self-join row t with row t+H across sessions.

Author: Anh Duc Le  (feature-store scaffolding co-developed with Claude.)
"""

import json
import math
import os
import time
from typing import Any, Optional


def _finite(obj: Any) -> Any:
    """
    Recursively replace non-finite floats (NaN / +-inf) with None.

    json.dumps() emits bare NaN/Infinity tokens by default, which are invalid JSON
    and choke strict readers (pandas, jq, other languages). The regime pipeline
    produces NaN (lambda when there is no cycle) and inf (strain when the graph is
    split), so this keeps every line valid JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def default_dir() -> str:
    """<repo root>/data -- created if missing."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "data")
    os.makedirs(d, exist_ok=True)
    return d


def default_path() -> str:
    """data/regime_YYYYmmdd_HHMMSS.jsonl -- unique per run."""
    return os.path.join(default_dir(), f"regime_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


class FeatureStore:
    """One JSONL file; append one line per tick, flushed so a Ctrl-C keeps the data."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_path()
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.rows = 0

    def write_meta(self, meta: dict) -> None:
        """
        Write a one-off sidecar `<path>.meta.json` describing HOW this run was gathered:
        the per-venue fee table, transfer cost, quote/stale windows, depth thresholds,
        stress params, venue list. This is the half that makes cost an OFFLINE knob --
        the JSONL holds fee-free raw metrics (lam_raw), and this file records the
        assumptions, so a labelling step can re-derive net λ / re-threshold STRESSED
        under a different fee WITHOUT re-gathering. Best-effort; never fatal.
        """
        try:
            with open(self.path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump(_finite(meta), fh, indent=2, default=str)
        except Exception:
            pass

    def append(self, row: dict) -> None:
        """Write one sanitised JSON object as a line and flush it to disk."""
        self._fh.write(json.dumps(_finite(row), separators=(",", ":")) + "\n")
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    # Context-manager sugar so callers can `with FeatureStore(...) as s:` if they want.
    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
