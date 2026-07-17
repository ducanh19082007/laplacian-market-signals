"""
feature_store.label -- turn raw per-tick logs into ML-ready rows with FORWARD labels.

The store writes only what is known at each tick. A *label* is what happens H seconds
LATER, so it can only be computed offline, after the fact, by joining each row t with
the row at t+H. That join is this file's whole job.

Run (globs are fine; sessions are concatenated and sorted by absolute ts):
    python -m feature_store.label "data/*.jsonl" --horizon 30 --out data/labeled.jsonl

For each input row it adds a `label` object:
    future_regime        regime H seconds later (the row at t+H)
    d_fiedler            fiedler(t+H) - fiedler(t)      (connectivity drift)
    d_lam                lam(t+H) - lam(t)              (arb-intensity drift; None if either NaN)
    frag_within_h        did ANY tick in (t, t+H] go FRAGMENTING?   (early-warning target)
    stress_within_h      did ANY tick in (t, t+H] go STRESSED?
Rows without a full H-second future (the tail of each run) are dropped -- they have no
label yet. Pure stdlib on purpose, so it runs with no extra deps; for real modelling
load the output with pandas: pd.read_json("data/labeled.jsonl", lines=True).

Author: Anh Duc Le  (co-developed with Claude.)
"""

import argparse
import bisect
import glob
import json
from collections import deque
from typing import List, Optional

FRAGMENTING = "FRAGMENTING"
STRESSED = "STRESSED"


def load_rows(patterns: List[str]) -> List[dict]:
    """Read every JSONL line across all glob patterns; skip torn/blank lines; sort by ts."""
    rows: List[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # torn final line of a killed run
                    if "ts" in r:
                        rows.append(r)
    rows.sort(key=lambda r: r["ts"])
    return rows


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return (a - b) if (a is not None and b is not None) else None


def add_rolling_regime(rows: List[dict], window: int = 240, q: float = 0.90) -> List[dict]:
    """
    Attach a FEE-INDEPENDENT stress flag `stress_raw` to each row, in place.

    This is the answer to "won't realistic fees leave me with no STRESSED examples?".
    The live `regime` field is net-of-fee, so on an efficient tape it is EFFICIENT
    almost always -- and making fees more accurate only makes that worse. So we do NOT
    label off it. Instead:

        stress_raw(t)  <=>  lam_raw(t) > q-quantile of lam_raw over the trailing `window`

    lam_raw is the FEE-FREE arb intensity (pure market dislocation), and the threshold
    is a quantile of the market's OWN recent distribution -- so ~(1 - q) of ticks flag
    STRESSED *by construction*, no matter how large the fees are. Fees decide whether a
    detected cycle is executable; they do NOT get a vote on what counts as a dislocation.

    Warm-up ticks (before the window has enough history) are left False. Ticks with no
    lam_raw (no cycle that instant) can't be stressed and don't feed the baseline.
    """
    buf: deque = deque(maxlen=window)
    warm = max(10, window // 4)
    for r in rows:
        lam = r.get("lam_raw")
        if lam is None:
            lam = r.get("lam")                         # runs predating lam_raw fall back to net
        stressed = False
        if lam is not None and len(buf) >= warm:
            ordered = sorted(buf)
            k = min(len(ordered) - 1, int(q * len(ordered)))
            stressed = lam > ordered[k]
        r["stress_raw"] = stressed
        if lam is not None:
            buf.append(lam)
    return rows


def label_rows(rows: List[dict], horizon: float, max_gap: float = 2.0) -> List[dict]:
    """
    Attach a forward-looking `label` to each row that has a full H-second future.

    Rows are globally sorted by ts so multiple sessions concatenate -- but a row is
    labelled ONLY if the tick nearest t+H actually lands near t+H (within max_gap).
    That drops the tail of every session (no real future) and, crucially, refuses to
    pair the END of one session with the START of the next across a multi-hour gap,
    which would otherwise fabricate a bogus "H-seconds-later" label. Raise max_gap if
    you sample slower than ~2s/tick.
    """
    ts = [r["ts"] for r in rows]
    out: List[dict] = []
    for i, r in enumerate(rows):
        target = r["ts"] + horizon
        if ts[-1] < target:
            break                                     # no later row exists anywhere
        j = bisect.bisect_left(ts, target, lo=i)      # first row at/after t+H
        if ts[j] - target > max_gap:                  # nearest future is across a data gap
            continue                                  # -> no valid H-second future; skip (not break)
        fut = rows[j]
        window = rows[i + 1:j + 1]                     # ticks in (t, t+H]
        r = dict(r)
        r["label"] = {
            "horizon": horizon,
            "future_regime": fut.get("regime"),
            "d_fiedler": _sub(fut.get("fiedler"), r.get("fiedler")),
            "d_lam": _sub(fut.get("lam"), r.get("lam")),
            "frag_within_h": any(w.get("regime") == FRAGMENTING for w in window),
            "stress_within_h": any(w.get("regime") == STRESSED for w in window),
            # fee-INDEPENDENT early-warning target (see add_rolling_regime) -- this is the
            # one to actually train on; the net-of-fee stress_within_h above is near-empty
            # on an efficient tape and only there for comparison.
            "stress_raw_within_h": any(w.get("stress_raw") for w in window),
        }
        out.append(r)
    return out


def summarize(labeled: List[dict]) -> None:
    n = len(labeled)
    if not n:
        print("no labeled rows (need at least one full horizon of data).")
        return
    frag = sum(1 for r in labeled if r["label"]["frag_within_h"])
    stress = sum(1 for r in labeled if r["label"]["stress_within_h"])
    stress_raw = sum(1 for r in labeled if r["label"]["stress_raw_within_h"])
    now_regimes: dict = {}
    for r in labeled:
        now_regimes[r.get("regime")] = now_regimes.get(r.get("regime"), 0) + 1
    print(f"labeled rows              : {n}")
    print(f"  regime now (counts)     : {now_regimes}")
    print(f"  fragment within H       : {frag}  ({100*frag/n:.1f}%)")
    print(f"  stress (net) within H   : {stress}  ({100*stress/n:.1f}%)")
    print(f"  stress_raw within H     : {stress_raw}  ({100*stress_raw/n:.1f}%)   <- fee-independent; train on this")


def main() -> None:
    ap = argparse.ArgumentParser(description="Add forward labels to feature-store JSONL logs.")
    ap.add_argument("patterns", nargs="+", help="one or more globs, e.g. 'data/*.jsonl'")
    ap.add_argument("--horizon", type=float, default=30.0, help="seconds ahead to look for the label")
    ap.add_argument("--max-gap", type=float, default=2.0,
                    help="skip a row if the tick nearest t+H is more than this many seconds off "
                         "(drops session-boundary rows; raise it if you sample slower than ~2s)")
    ap.add_argument("--out", default="data/labeled.jsonl", help="output JSONL path")
    ap.add_argument("--stress-window", type=int, default=240,
                    help="trailing ticks for the fee-free STRESSED baseline (rolling quantile of lam_raw)")
    ap.add_argument("--stress-quantile", type=float, default=0.90,
                    help="lam_raw above this quantile of the trailing window == STRESSED_raw "
                         "(so ~(1-q) of ticks flag, regardless of fees)")
    args = ap.parse_args()

    rows = load_rows(args.patterns)
    print(f"loaded {len(rows)} raw rows from {args.patterns}")
    if len(rows) < 2:
        print("need at least 2 rows to compute a forward label -- gather more data first.")
        return

    # Fee-INDEPENDENT stress flag before we look forward, so the H-second window can ask
    # "did a dislocation happen soon?" without fees getting a vote (the STRESSED-starvation
    # worry). Must run on the ts-sorted rows so the rolling baseline is chronological.
    add_rolling_regime(rows, args.stress_window, args.stress_quantile)

    labeled = label_rows(rows, args.horizon, args.max_gap)
    if not labeled:
        # Not a bug: a row's label is the state H seconds LATER, so a log shorter than
        # the horizon has no fully-labelled rows. Say so, with an actionable fix.
        span = rows[-1]["ts"] - rows[0]["ts"]
        hint = max(1, int(span / 3))
        print(f"0 rows labeled: this log spans only {span:.1f}s, but --horizon is {args.horizon:g}s, "
              f"so no row has a full {args.horizon:g}s future yet.")
        print(f"  -> gather at least ~{args.horizon:g}s of data, or retry on what you have with "
              f"a shorter --horizon {hint}.")
        return

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in labeled:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(labeled)} labeled rows -> {args.out}  (horizon {args.horizon:g}s)")
    summarize(labeled)


if __name__ == "__main__":
    main()
