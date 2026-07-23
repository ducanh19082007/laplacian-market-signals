"""
L5 -- ANOMALY FOREST v2  (PHYSICS-INFORMED, PER-REGIME supervised detector)
===============================================================================

anomaly_forest.py scores ONE generic "weirdness" number that blends FRAGMENTING + novelty
+ stress together, so asking it to name a SPECIFIC regime tops out around F1 0.45. This v2
takes the opposite, physics-first stance, from the empirical fact that the two named
non-efficient regimes live on ORTHOGONAL axes of the same graph:

    STRESSED     arb intensity lambda is HIGH vs its own recent baseline.
                 (median lam_raw ~2.6x normal; fiedler roughly unchanged.)
    FRAGMENTING  algebraic connectivity lambda_2 (Fiedler value) COLLAPSES toward 0 as the
                 venue graph splits; with no spanning cycle, lam_raw also falls to ~0.

So instead of one detector we train TWO specialised supervised heads, each seeing ONLY the
feature block that physically DEFINES its regime -- a deliberate inductive bias, not a
data-driven guess:

    STRESS head  <- LAM block   : lam_raw, lam_dev, lam_z, lam_std        (how high is lambda?)
    FRAG   head  <- CONN block  : fiedler, fiedler_std, d_fiedler, n_components  (has lambda_2 collapsed?)

Each head is a HistGradientBoosting classifier (handles NaN warm-up rows natively, class-
weight balances the ~90/10 split), trained on its fee-free label -- stress_raw for STRESS,
regime==FRAGMENTING for FRAG -- with the probability cutoff picked to MAXIMISE F1 on TRAIN.
The two heads are then combined into a 3-class regime call (EFFICIENT / STRESSED /
FRAGMENTING); when both fire, the head with the larger margin over its own threshold wins
(rare -- the axes barely overlap).

HONEST CAVEAT (same as v1): each head is PARTLY CIRCULAR -- its label is defined off the very
block it reads (stress_raw off lam, FRAGMENTING off connectivity). The payoff is NOT
discovering a hidden signal; it is (1) a calibrated per-regime PROBABILITY instead of a hard
rule, (2) far cleaner predicted-vs-real counts per regime than the generic forest, and (3) a
clean two-head scaffold to later re-point at a FORWARD label (regime H seconds ahead) -- which
is where the ML genuinely earns its keep. Evaluation is the same honest chronological
train/test split as v1: metrics on held-out sessions the model never saw.

PRIORITISING STRESSED
Two ways, and the data decides which is real:
  * NOWCAST, recall-first (WORKS): detect stress the instant it happens and tune the STRESS
    head to catch as much as possible (--stress-recall 0.9 or --stress-beta 2). Nowcast stress
    AUC ~0.98, so you can push recall high cheaply. This is the recommended way to prioritise it.
  * FORECAST, --forecast (STRESS DOESN'T FORECAST): re-point the heads at "what STATE ~H sec
    from now?" and OR them into one 'anomaly ahead' alert. Honest finding: STRESS-ahead AUC is
    ~0.50 at EVERY horizon (even 1s) -- an arbitrage dislocation is a surprise, not predictable
    from graph features. Only FRAGMENTING has real lead time (AUC ~0.69 at ~5s), so the forward
    alert's power comes from FRAG, not stress. The mode prints this caveat when a head is ~random.

Run (from the repo root):
    python "L5_MachineLearningImplementation/anomaly_forest_2.py"            # nowcast: train + eval + plots
    python "L5_MachineLearningImplementation/anomaly_forest_2.py" --no-plots
    python "L5_MachineLearningImplementation/anomaly_forest_2.py" --stress-recall 0.9  # NOWCAST, catch >=90% stress
    python "L5_MachineLearningImplementation/anomaly_forest_2.py" --forecast --horizon 5  # forward alert (frag-driven)

Author: Anh Duc Le
L5 anomaly forest v2 (physics-informed per-regime detector) co-developed with Claude
(Opus 4.8, 1M context).
"""

import argparse
import bisect
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support)

# Reuse v1's data pipeline and constants so the two models are judged on IDENTICAL rows.
from anomaly_forest import (EFFICIENT, FRAGMENTING, STRESSED, WARM, DEFAULT_TEST, DEFAULT_TRAIN,
                            _auc, _build, _clean_mask, feature_matrix)
from feature_store.label import add_rolling_regime, load_rows

# Physical feature blocks (see module docstring). Selected BY NAME so they stay correct even
# if feature_matrix's column order changes.
LAM_BLOCK = ["lam_raw", "lam_dev", "lam_z", "lam_std"]                 # STRESSED lives here
CONN_BLOCK = ["fiedler", "fiedler_std", "d_fiedler", "n_components"]   # FRAGMENTING lives here
DEFAULT_MODEL_V2 = os.path.join(_ROOT, "models", "anomaly_forest_2.joblib")

REGIMES = [EFFICIENT, STRESSED, FRAGMENTING]
_RGB = {EFFICIENT: "#2ca02c", STRESSED: "#ff7f0e", FRAGMENTING: "#d62728"}


# ===========================================================================
# One specialised, self-thresholding regime head
# ===========================================================================
class RegimeHead:
    """A HistGradientBoosting classifier restricted to ONE physical feature block, with the
    probability cutoff that maximises F1 on TRAIN bundled in. HIGHER proba = MORE that regime."""

    def __init__(self, name, block):
        self.name = name              # STRESSED / FRAGMENTING
        self.block = list(block)      # feature names this head is allowed to see
        self.cols = None              # their column indices in the full matrix
        self.model = None
        self.threshold_ = 0.5
        self.meta = {}

    def fit(self, X, y, feature_names, *, random_state=42, max_iter=400):
        self.cols = [feature_names.index(f) for f in self.block]
        self.model = HistGradientBoostingClassifier(
            random_state=random_state, class_weight="balanced", learning_rate=0.08,
            max_iter=max_iter, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False).fit(X[:, self.cols], y.astype(int))
        self.meta = dict(n_train=len(X), n_pos=int(y.sum()), block=self.block)
        return self

    def proba(self, X):
        return self.model.predict_proba(X[:, self.cols])[:, 1]

    def pick_threshold(self, X, y, beta=1.0, recall_target=None):
        """Choose the probability cutoff on the given (train) rows.

        beta=1 -> MAX F1 (balanced). beta>1 weights RECALL beta-times more than precision
        (beta=2 is the usual "catch it" setting). If recall_target is given it overrides beta:
        take the highest-precision cutoff that still reaches that recall -- the honest way to
        say "I must catch >= X% of these, spend the least precision doing it"."""
        if y.min() == y.max():
            self.threshold_ = 0.5
            return self.threshold_
        p = self.proba(X)
        prec, rec, th = precision_recall_curve(y, p)
        if recall_target is not None:
            ok = np.where(rec[:-1] >= recall_target)[0]
            j = ok[np.argmax(prec[ok])] if len(ok) else int(np.argmax(rec[:-1]))
        else:
            b2 = beta * beta
            fb = np.divide((1 + b2) * prec * rec, b2 * prec + rec,
                           out=np.zeros_like(prec), where=(b2 * prec + rec) > 0)
            j = max(0, int(np.argmax(fb[:-1])))
        self.threshold_ = float(th[j])
        return self.threshold_

    def predict(self, X):
        return self.proba(X) >= self.threshold_

    def fit_calibrated(self, X, y, feature_names, beta=1.0, recall_target=None, val_frac=0.25):
        """Fit + pick the cutoff HONESTLY: train the model on the first (1-val_frac) of the
        (chronological) rows, then choose the threshold on the held-out TAIL -- out-of-sample.

        In-sample probabilities are overfit-confident, so a recall target picked on them
        undershoots badly on unseen data (a 'catch 90%' cutoff can deliver ~70%). Calibrating
        on a genuine hold-out makes the target GENERALISE. The model is kept as the fit-part
        model (not refit on all rows) so its probabilities still match the chosen threshold."""
        self.cols = [feature_names.index(f) for f in self.block]
        Xb = X[:, self.cols]
        n = len(X); cut = max(WARM, int(n * (1 - val_frac)))
        self.model = HistGradientBoostingClassifier(
            random_state=42, class_weight="balanced", learning_rate=0.08, max_iter=400,
            max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False).fit(
                Xb[:cut], y[:cut].astype(int))
        # pick_threshold uses self.proba -> the TAIL rows are out-of-sample for this model.
        pr, rc, th = precision_recall_curve(y[cut:], self.model.predict_proba(Xb[cut:])[:, 1])
        if recall_target is not None:
            ok = np.where(rc[:-1] >= recall_target)[0]
            j = ok[np.argmax(pr[ok])] if len(ok) else int(np.argmax(rc[:-1]))
        else:
            b2 = beta * beta
            fb = np.divide((1 + b2) * pr * rc, b2 * pr + rc, out=np.zeros_like(pr), where=(b2 * pr + rc) > 0)
            j = max(0, int(np.argmax(fb[:-1])))
        self.threshold_ = float(th[j])
        self.meta = dict(n_fit=cut, n_val=n - cut, block=self.block, calibrated=True)
        return self


# ===========================================================================
# The two-head detector
# ===========================================================================
class RegimeDetectorV2:
    """STRESS head (lam block) + FRAG head (connectivity block) -> a 3-class regime call."""

    def __init__(self):
        self.stress = RegimeHead(STRESSED, LAM_BLOCK)
        self.frag = RegimeHead(FRAGMENTING, CONN_BLOCK)
        self.features = None

    def fit(self, X, feature_names, sr, frag, stress_beta=1.0, stress_recall=None):
        """stress_beta>1 (or stress_recall) tunes the STRESS head RECALL-FIRST -- prioritise
        catching stress, at the cost of precision. FRAG head stays at max-F1. A recall TARGET
        is calibrated on a hold-out (fit_calibrated) so it actually generalises to unseen data."""
        self.features = list(feature_names)
        self.frag.fit(X, frag, self.features)
        self.frag.pick_threshold(X, frag)
        if stress_recall is not None:
            self.stress.fit_calibrated(X, sr, self.features, recall_target=stress_recall)
        else:
            self.stress.fit(X, sr, self.features)
            self.stress.pick_threshold(X, sr, beta=stress_beta)
        return self

    def predict_regime(self, X):
        """Return (regime[], stress_proba[], frag_proba[]).  Both-fire ties break to the head
        with the larger margin over its own threshold (rare: the axes barely overlap)."""
        ps, pf = self.stress.proba(X), self.frag.proba(X)
        fire_s, fire_f = ps >= self.stress.threshold_, pf >= self.frag.threshold_
        out = np.full(len(X), EFFICIENT, dtype=object)
        out[fire_s] = STRESSED
        out[fire_f] = FRAGMENTING                                   # provisional; resolve ties next
        both = fire_s & fire_f
        margin_s, margin_f = ps - self.stress.threshold_, pf - self.frag.threshold_
        out[both] = np.where(margin_f[both] >= margin_s[both], FRAGMENTING, STRESSED)
        return out, ps, pf

    def save(self, path=DEFAULT_MODEL_V2):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=DEFAULT_MODEL_V2) -> "RegimeDetectorV2":
        return joblib.load(path)


def true_regime(sr, frag):
    """3-class ground truth from the two fee-free labels. FRAGMENTING wins ties (a structural
    break is the more severe state, and it drives lam_raw to ~0 so stress_raw rarely coexists)."""
    out = np.full(len(sr), EFFICIENT, dtype=object)
    out[sr] = STRESSED
    out[frag] = FRAGMENTING
    return out


# ===========================================================================
# FORWARD-labelled data for EARLY WARNING ("what STATE will it be in, H sec from now?")
# ===========================================================================
def _forward_state_at_h(rows, horizon, max_gap=3.0):
    """
    Forward target = the STATE at the tick ~`horizon` seconds AHEAD (a point prediction).

    NOT "any anomalous tick in the next H sec": stress_raw fires ~10% of ticks and flickers, so
    "any stress within a 30s window" compounds to 1-(1-0.1)^60 ~= 99.8% -- degenerate, a constant
    "yes" scores ~99.7%. That compounding hits ANY window target, even rare events. A POINT
    prediction (is the tick H sec from now stressed?) keeps the base rate at the per-tick ~10%,
    so it is a real forecast with a measurable ROC. Causal: target reads a FUTURE tick, features
    only PAST -- no leakage. Rows with no tick near t+H (session tail / a data gap > max_gap) are
    marked invalid and dropped.

    Returns (stress_ahead, frag_ahead, valid) as bool arrays over `rows`.
    """
    ts = [r["ts"] for r in rows]
    s_now = np.array([bool(r.get("stress_raw")) for r in rows])
    f_now = np.array([r.get("regime") == FRAGMENTING for r in rows])
    n = len(rows)
    sa = np.zeros(n, dtype=bool); fa = np.zeros(n, dtype=bool); valid = np.zeros(n, dtype=bool)
    for i in range(n):
        target = ts[i] + horizon
        if ts[-1] < target:                                 # no H-second future left in this session
            continue
        j = bisect.bisect_left(ts, target, lo=i)            # first tick at/after t+H
        if ts[j] - target > max_gap:                        # nearest future tick is across a data gap
            continue
        valid[i] = True
        sa[i] = s_now[j]; fa[i] = f_now[j]
    return sa, fa, valid


def _build_forward(pattern, args):
    """
    Per session: attach fee-free stress_raw, derive the state-at-t+H targets
    (_forward_state_at_h), compute CAUSAL features, keep only clean rows with a valid future.
    Returns (X, stress_ahead, frag_ahead, rows, segments, feature_names, n_files).
    """
    files = sorted(glob.glob(pattern))
    Xs, sa, fa, rows_all, seg = [], [], [], [], []
    names = None
    for path in files:
        rows = load_rows([path])
        if len(rows) < WARM:
            continue
        add_rolling_regime(rows, args.stress_window, args.stress_q)     # adds stress_raw
        s_ahead, f_ahead, valid = _forward_state_at_h(rows, args.horizon)
        keep = _clean_mask(rows, args.min_nodes, args.warmup) & valid
        rows_c = [r for r, k in zip(rows, keep) if k]
        if not rows_c:
            continue
        X, names = feature_matrix(rows_c)
        start = len(rows_all)
        Xs.append(X); sa.append(s_ahead[keep]); fa.append(f_ahead[keep])
        rows_all.extend(rows_c); seg.append((start, len(rows_all)))
    if not Xs:
        raise ValueError(f"no forward-labelled rows under {pattern} -- is each session longer "
                         f"than --horizon {args.horizon:g}s?")
    return (np.vstack(Xs), np.concatenate(sa), np.concatenate(fa), rows_all, seg, names, len(files))


# ===========================================================================
# Report
# ===========================================================================
def _head_line(name, y_true, proba, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    real, predicted, caught = int(y_true.sum()), int(pred.sum()), int((pred & y_true).sum())
    auc = _auc(y_true, proba)
    ap = average_precision_score(y_true, proba) if y_true.min() != y_true.max() else float("nan")
    print(f"\n  [{name} head]  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}   "
          f"ROC-AUC {auc:.3f}  PR-AUC {ap:.3f}")
    print(f"      confusion (TN {tn}  FP {fp}  FN {fn}  TP {tp})")
    print(f"      predicted vs real : {predicted} predicted, {real} real, {caught} caught "
          f"({(100*caught/real if real else float('nan')):.1f}% recall)")


def _report(det, Xte, srte, fragte):
    reg_pred, ps, pf = det.predict_regime(Xte)
    reg_true = true_regime(srte, fragte)
    n = len(reg_true)
    print(f"\n=== ANOMALY FOREST v2 -- held-out TEST  ({n} ticks) ===")
    print(f"  STRESS head reads {det.stress.block}  (cutoff prob>={det.stress.threshold_:.3f})")
    print(f"  FRAG   head reads {det.frag.block}  (cutoff prob>={det.frag.threshold_:.3f})")

    _head_line(STRESSED, srte, ps, ps >= det.stress.threshold_)
    _head_line(FRAGMENTING, fragte, pf, pf >= det.frag.threshold_)

    # Combined 3-class regime call.
    print("\n  --- combined 3-class regime ---")
    acc = accuracy_score(reg_true, reg_pred)
    bal = balanced_accuracy_score(reg_true, reg_pred)
    print(f"  overall accuracy {acc:.3f}   balanced-acc {bal:.3f}")
    cm = confusion_matrix(reg_true, reg_pred, labels=REGIMES)
    print(f"    {'true \\ pred':<14}" + "".join(f"{r:>13}" for r in REGIMES))
    for i, r in enumerate(REGIMES):
        print(f"    {r:<14}" + "".join(f"{cm[i, j]:>13d}" for j in range(len(REGIMES))))
    print("\n  per-regime PREDICTED vs REAL (combined call):")
    print(f"    {'regime':<14}{'real':>9}{'predicted':>11}{'recall':>9}")
    for r in REGIMES:
        real = int((reg_true == r).sum())
        predicted = int((reg_pred == r).sum())
        caught = int(((reg_pred == r) & (reg_true == r)).sum())
        rec = f"{100*caught/real:.1f}%" if real else "n/a"
        print(f"    {r:<14}{real:>9}{predicted:>11}{rec:>9}")
    print("  (each head is PARTLY CIRCULAR: its label is defined off the block it reads -- "
          "the win is calibrated per-regime probabilities + clean counts, see module docstring.)")
    return reg_pred, reg_true, ps, pf


def _forecast_report(H, s_head, f_head, Xte, s_te, f_te):
    """Held-out report for the EARLY-WARNING model: one 'anomaly ahead' alert (STRESS-ahead OR
    FRAG-ahead), tuned so STRESSED is caught aggressively and FRAGMENTING is left at max-F1."""
    ps, pf = s_head.proba(Xte), f_head.proba(Xte)
    pred_s, pred_f = ps >= s_head.threshold_, pf >= f_head.threshold_
    alert = pred_s | pred_f
    true_any = s_te | f_te
    acc = accuracy_score(true_any, alert)
    prec, rec, f1, _ = precision_recall_fscore_support(true_any, alert, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_any, alert, labels=[False, True]).ravel()

    print(f"\n=== EARLY-WARNING -- 'anomaly expected ~{H:g}s from now' -- held-out TEST ===")
    print("  output is ONE alert (STRESS-ahead OR FRAG-ahead); we do NOT tell the user which.")
    print(f"  alert  : accuracy {acc:.3f}  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  "
          f"(prevalence {true_any.mean():.3f})")
    print(f"  confusion (TN {tn}  FP {fp}  FN {fn}  TP {tp})   alert fires on {100*alert.mean():.1f}% of ticks")
    print("\n  what the alert catches, BY SOURCE (the priority split):")
    for name, lab, thr, p in [("STRESS-ahead", s_te, s_head.threshold_, ps),
                              ("FRAG-ahead  ", f_te, f_head.threshold_, pf)]:
        real = int(lab.sum()); caught = int((alert & lab).sum())
        pr = f"{100*caught/real:.1f}%" if real else "n/a"
        auc = _auc(lab, p)
        # An AUC near 0.5 means the head has NO forward signal -- the label is not forecastable
        # at this horizon, so any "recall" it shows is just the alert firing indiscriminately.
        if auc < 0.55:
            tag = "  <- NO forward signal (AUC~0.5): NOT forecastable at this horizon"
        elif name.startswith("STRESS"):
            tag = "  <- prioritised (recall-first)"
        else:
            tag = "  <- left at max-F1"
        print(f"    {name}: caught {caught}/{real} ({pr})   head cutoff {thr:.3f}  ROC-AUC {auc:.3f}{tag}")
    print(f"  lead time: a fired alert says the anomaly is expected ~{H:g}s out.")
    print("  NOTE: STRESS is typically UNFORECASTABLE (AUC~0.5 at every horizon) -- a dislocation is a"
          " surprise.\n        To prioritise stress, DETECT it instantly (nowcast) with --stress-recall,"
          " not forecast it.\n        FRAGMENTING is the regime with real lead time; the alert's forward"
          " power comes from it.")
    return ps, pf, alert, true_any


# ===========================================================================
# Plots
# ===========================================================================
def _plots(det, Xte, srte, fragte, reg_pred, reg_true, ps, pf, te_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = Xte[:, det.features.index("lam_raw")]
    fie = Xte[:, det.features.index("fiedler")]

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))

    # (0,0) THE physics: lam vs Fiedler, coloured by TRUE regime. STRESSED rides high on lam,
    # FRAGMENTING hugs the fiedler~0 wall. This is the whole thesis of v2 in one panel.
    a = ax[0, 0]
    lam_plot = np.clip(lam, 1e-7, None)
    for r in REGIMES:
        m = reg_true == r
        a.scatter(fie[m], lam_plot[m], s=6, alpha=0.35, color=_RGB[r], label=r)
    a.set_yscale("log")
    a.set(title="the physics: arb intensity lambda vs connectivity lambda_2 (true regime)",
          xlabel="fiedler  lambda_2  (-> 0 = FRAGMENTING)", ylabel="lam_raw  lambda  (high = STRESSED)")
    a.legend(fontsize=8, markerscale=2)

    # (0,1) STRESS head PR curve.
    a = ax[0, 1]
    if srte.min() != srte.max():
        pr, rc, _ = precision_recall_curve(srte, ps)
        a.plot(rc, pr, color=_RGB[STRESSED], label=f"AP {average_precision_score(srte, ps):.3f}")
        a.axhline(srte.mean(), ls=":", color="grey", label=f"baseline {srte.mean():.2f}")
    a.set(title="STRESS head Precision-Recall (test)", xlabel="recall", ylabel="precision"); a.legend(fontsize=8)

    # (1,0) FRAG head PR curve.
    a = ax[1, 0]
    if fragte.min() != fragte.max():
        pr, rc, _ = precision_recall_curve(fragte, pf)
        a.plot(rc, pr, color=_RGB[FRAGMENTING], label=f"AP {average_precision_score(fragte, pf):.3f}")
        a.axhline(fragte.mean(), ls=":", color="grey", label=f"baseline {fragte.mean():.2f}")
    a.set(title="FRAG head Precision-Recall (test)", xlabel="recall", ylabel="precision"); a.legend(fontsize=8)

    # (1,1) per-regime predicted vs real bars.
    a = ax[1, 1]
    reals = [int((reg_true == r).sum()) for r in REGIMES]
    preds = [int((reg_pred == r).sum()) for r in REGIMES]
    x = np.arange(len(REGIMES)); bw = 0.38
    b1 = a.bar(x - bw / 2, reals, bw, color="#4c72b0", label="real")
    b2 = a.bar(x + bw / 2, preds, bw, color="#dd8452", label="predicted")
    a.bar_label(b1, fontsize=8); a.bar_label(b2, fontsize=8)
    a.set_yscale("log")
    a.set(title="per-regime real vs predicted (combined call)", ylabel="ticks (log)",
          xticks=x); a.set_xticklabels(REGIMES, fontsize=9); a.legend(fontsize=9)

    fig.suptitle("L5 Anomaly Forest v2 -- physics-informed per-regime detector (held-out test)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_ROOT, "anomaly_forest2_eval.png")
    fig.savefig(out, dpi=120)
    print(f"\n[plots] wrote {out}")


def _forecast_plot(H, s_head, s_te, f_te, ps, pf, alert, true_any):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # STRESS-ahead PR (the prioritised head) with its operating point marked.
    a = ax[0]
    if s_te.min() != s_te.max():
        pr, rc, _ = precision_recall_curve(s_te, ps)
        a.plot(rc, pr, color=_RGB[STRESSED], label=f"AP {average_precision_score(s_te, ps):.3f}")
        pred = ps >= s_head.threshold_
        rec_op = (pred & s_te).sum() / max(1, s_te.sum())
        prec_op = (pred & s_te).sum() / max(1, pred.sum())
        a.scatter([rec_op], [prec_op], color="black", zorder=5, label=f"cutoff (rec {rec_op:.2f})")
        a.axhline(s_te.mean(), ls=":", color="grey")
    a.set(title=f"STRESS-ahead PR (next {H:g}s) -- PRIORITISED", xlabel="recall", ylabel="precision")
    a.legend(fontsize=8)
    # FRAG-ahead PR.
    a = ax[1]
    if f_te.min() != f_te.max():
        pr, rc, _ = precision_recall_curve(f_te, pf)
        a.plot(rc, pr, color=_RGB[FRAGMENTING], label=f"AP {average_precision_score(f_te, pf):.3f}")
        a.axhline(f_te.mean(), ls=":", color="grey")
    a.set(title=f"FRAG-ahead PR (next {H:g}s) -- max-F1", xlabel="recall", ylabel="precision")
    a.legend(fontsize=8)
    # Unified alert confusion.
    a = ax[2]
    cm = confusion_matrix(true_any, alert, labels=[False, True])
    a.imshow(cm, cmap="Purples")
    a.set(title=f"'anomaly within {H:g}s' alert (test)", xticks=[0, 1], yticks=[0, 1],
          xticklabels=["no alert", "ALERT"], yticklabels=["calm ahead", "anomaly ahead"])
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.suptitle(f"L5 Anomaly Forest v2 -- EARLY WARNING (anomaly within {H:g}s, STRESS prioritised)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(_ROOT, "anomaly_forest2_forecast.png")
    fig.savefig(out, dpi=120)
    print(f"[plots] wrote {out}")


# ===========================================================================
# Orchestration
# ===========================================================================
def run_forecast(args):
    """EARLY-WARNING mode: predict 'anomaly within H sec' as ONE alert, STRESS prioritised."""
    Xtr, s_tr, f_tr, tr_rows, seg_tr, names, ntr = _build_forward(args.train_data, args)
    Xte, s_te, f_te, te_rows, seg_te, names2, nte = _build_forward(args.test_data, args)
    print(f"[train] {len(Xtr)} forward-labelled rows from {ntr} session(s)  "
          f"(stress-ahead {int(s_tr.sum())}, frag-ahead {int(f_tr.sum())})   horizon {args.horizon:g}s")
    print(f"[test ] {len(Xte)} forward-labelled rows from {nte} session(s)  "
          f"(stress-ahead {int(s_te.sum())}, frag-ahead {int(f_te.sum())})")

    idx = slice(None, None, args.stride)
    # Both forward heads read ALL features (forecasting isn't definitional, so the physics-block
    # restriction is dropped -- e.g. connectivity drift NOW can foreshadow a stress spike SOON).
    s_head = RegimeHead("STRESS-ahead", names).fit(Xtr[idx], s_tr[idx], names)
    f_head = RegimeHead("FRAG-ahead", names).fit(Xtr[idx], f_tr[idx], names)
    # STRESS prioritised: recall-target if given, else F-beta (beta>1 favours recall). FRAG at F1.
    s_head.pick_threshold(Xtr, s_tr, beta=args.stress_beta, recall_target=args.stress_recall)
    f_head.pick_threshold(Xtr, f_tr, beta=1.0)
    tuned = (f"recall-target {args.stress_recall:g}" if args.stress_recall is not None
             else f"F{args.stress_beta:g}")
    print(f"[fit  ] {len(Xtr[idx])} rows; STRESS head tuned {tuned} (prioritised), FRAG head max-F1")

    ps, pf, alert, true_any = _forecast_report(args.horizon, s_head, f_head, Xte, s_te, f_te)
    if not args.no_plots:
        _forecast_plot(args.horizon, s_head, s_te, f_te, ps, pf, alert, true_any)


def run(args):
    if args.forecast:
        return run_forecast(args)
    Xtr, ytr, fragtr, srtr, tr_rows, seg_tr, names, ntr = _build(args.train_data, args)
    Xte, yte, fragte, srte, te_rows, seg_te, names_te, nte = _build(args.test_data, args)
    print(f"[train] {len(Xtr)} clean rows from {ntr} session(s)  "
          f"(stress {int(srtr.sum())}, frag {int(fragtr.sum())})")
    print(f"[test ] {len(Xte)} clean rows from {nte} session(s)  "
          f"(stress {int(srte.sum())}, frag {int(fragte.sum())})")

    Xs, srs, frs = Xtr[:: args.stride], srtr[:: args.stride], fragtr[:: args.stride]
    print(f"[fit  ] {len(Xs)} rows (stride={args.stride})")

    if args.no_train:
        det = RegimeDetectorV2.load(args.model)
        det.features = list(names)                                  # re-bind in case features changed
        print(f"[model] loaded {args.model}")
    else:
        det = RegimeDetectorV2().fit(Xs, names, srs, frs,
                                     stress_beta=args.stress_beta, stress_recall=args.stress_recall)
        if args.stress_recall is not None or args.stress_beta != 1.0:
            tuned = (f"recall-target {args.stress_recall:g}" if args.stress_recall is not None
                     else f"F{args.stress_beta:g}")
            print(f"[stress] NOWCAST STRESS head prioritised ({tuned}); cutoff prob>={det.stress.threshold_:.3f}")
        print(f"[model] saved -> {det.save(args.model)}")

    reg_pred, reg_true, ps, pf = _report(det, Xte, srte, fragte)
    if not args.no_plots:
        _plots(det, Xte, srte, fragte, reg_pred, reg_true, ps, pf, te_rows)


def main():
    ap = argparse.ArgumentParser(
        description="L5 anomaly forest v2 -- physics-informed per-regime supervised detector.")
    ap.add_argument("--no-train", action="store_true", help="reuse the saved v2 model")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--train-data", default=DEFAULT_TRAIN, dest="train_data")
    ap.add_argument("--test-data", default=DEFAULT_TEST, dest="test_data")
    ap.add_argument("--model", default=DEFAULT_MODEL_V2)
    ap.add_argument("--stride", type=int, default=1, help="thin training rows (try 4-8 for autocorrelation)")
    ap.add_argument("--min-nodes", type=int, default=10, dest="min_nodes")
    ap.add_argument("--warmup", type=float, default=120.0)
    ap.add_argument("--stress-window", type=int, default=240, dest="stress_window")
    ap.add_argument("--stress-q", type=float, default=0.90, dest="stress_q")
    # ---- EARLY-WARNING (forecast) mode ----
    ap.add_argument("--forecast", action="store_true",
                    help="EARLY-WARNING: emit ONE 'anomaly within H sec' alert (STRESS-ahead OR "
                         "FRAG-ahead), with STRESS prioritised. Uses forward labels, not nowcast.")
    ap.add_argument("--horizon", type=float, default=30.0, help="how many seconds ahead to warn (forecast mode)")
    ap.add_argument("--stress-beta", type=float, default=1.0, dest="stress_beta",
                    help="F-beta for the STRESS head (both modes): beta>1 favours RECALL -- catch more "
                         "stress at some precision cost. Default 1.0 = max-F1. Ignored if --stress-recall set.")
    ap.add_argument("--stress-recall", type=float, default=None, dest="stress_recall",
                    help="instead of F-beta, force the STRESS head to reach this recall (e.g. 0.90) "
                         "at the best precision that allows -- the direct way to prioritise catching stress")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
