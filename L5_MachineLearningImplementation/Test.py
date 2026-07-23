"""
Episode-aware test: is FRAGMENTING forecastable, or is 0.70 AUC a few-episode fluke?
Read-only. Uses the repo's own data + feature logic (feature_matrix validated == pyc).
"""
import bisect, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ducanh19082007/FOREX_farming")
from feature_store.label import load_rows, add_rolling_regime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
FRAG = "FRAGMENTING"
ROLL, LAG, WARM, EPS = 120, 20, 20, 1e-6
TRAIN = "/home/ducanh19082007/FOREX_farming/data/train/*.jsonl"
TEST  = "/home/ducanh19082007/FOREX_farming/data/test/*.jsonl"

import pandas as pd
def feature_matrix(rows):
    lam = pd.Series([r.get("lam_raw") if isinstance(r.get("lam_raw"),(int,float)) else np.nan
                     for r in rows], dtype=float).fillna(0.0)
    fie = pd.Series([r.get("fiedler") if isinstance(r.get("fiedler"),(int,float)) else np.nan
                     for r in rows], dtype=float)
    nc  = pd.Series([r.get("n_components", np.nan) for r in rows], dtype=float)
    lm, ls = lam.rolling(ROLL,min_periods=WARM).mean(), lam.rolling(ROLL,min_periods=WARM).std()
    dev = lam - lm
    df = pd.DataFrame({"lam_raw":lam,"fiedler":fie,"n_components":nc,"lam_dev":dev,
        "lam_z":dev/(ls+EPS),"lam_std":ls,"fiedler_std":fie.rolling(ROLL,min_periods=WARM).std(),
        "d_fiedler":fie-fie.shift(LAG)})
    return df.to_numpy(dtype=float)

def clean_mask(rows, min_nodes=10, warmup=120.0):
    return np.array([(r.get("n_nodes",0)>=min_nodes) and (r.get("t",1e18)>=warmup) for r in rows])

def forward_frag(rows, H, max_gap=3.0):
    ts = [r["ts"] for r in rows]
    f_now = np.array([r.get("regime")==FRAG for r in rows])
    n=len(rows); fa=np.zeros(n,bool); valid=np.zeros(n,bool)
    for i in range(n):
        tgt=ts[i]+H
        if ts[-1]<tgt: continue
        j=bisect.bisect_left(ts,tgt,lo=i)
        if ts[j]-tgt>max_gap: continue
        valid[i]=True; fa[i]=f_now[j]
    return fa, valid

def build(pattern, H):
    """Per-session forward-labelled clean rows. Returns X, y(frag-ahead), seg (per-session slices)."""
    Xs, ys, segs = [], [], []
    start=0
    for path in sorted(glob.glob(pattern)):
        rows=load_rows([path])
        if len(rows)<WARM: continue
        add_rolling_regime(rows,240,0.90)
        fa, valid = forward_frag(rows, H)
        keep = clean_mask(rows) & valid
        idx=np.where(keep)[0]
        if idx.size==0: continue
        rc=[rows[i] for i in idx]
        X=feature_matrix(rc); y=fa[idx]
        Xs.append(X); ys.append(y); segs.append((start,start+len(y))); start+=len(y)
    return np.vstack(Xs), np.concatenate(ys), segs

# ---- episode structure on the raw NOWCAST fragmentation truth (per test session) ----
def episode_stats():
    print("="*78); print("FRAGMENTATION EPISODE STRUCTURE (test sessions, nowcast truth)"); print("="*78)
    tot_ep=tot_frag=0; all_dur=[]; all_gap=[]; all_iat=[]
    for path in sorted(glob.glob(TEST)):
        rows=load_rows([path]); ts=np.array([r["ts"] for r in rows])
        frag=np.array([r.get("regime")==FRAG for r in rows])
        # contiguous runs of frag==True, merging runs separated by < 4 s (same event flickering)
        runs=[]; i=0; n=len(frag)
        while i<n:
            if frag[i]:
                j=i
                while j+1<n and frag[j+1]: j+=1
                runs.append([i,j]); i=j+1
            else: i+=1
        merged=[]
        for r in runs:
            if merged and ts[r[0]]-ts[merged[-1][1]] < 4.0: merged[-1][1]=r[1]
            else: merged.append(r)
        durs=[ts[b]-ts[a] for a,b in merged]
        starts=[ts[a] for a,b in merged]
        iats=list(np.diff(starts)) if len(starts)>1 else []
        tot_ep+=len(merged); tot_frag+=int(frag.sum())
        all_dur+=durs; all_iat+=iats
        span=ts[-1]-ts[0]
        print(f"  {os.path.basename(path):40s} span {span/60:5.1f}min  "
              f"frag ticks {int(frag.sum()):5d}  episodes {len(merged):3d}  "
              f"median dur {np.median(durs) if durs else 0:5.1f}s")
    print("-"*78)
    print(f"  TOTAL fragmentation episodes across all test sessions : {tot_ep}")
    print(f"  TOTAL fragmentation ticks                             : {tot_frag}")
    print(f"  => effective independent events ~ {tot_ep}, NOT {tot_frag} ticks")
    if all_dur:
        d=np.array(all_dur)
        print(f"  episode duration  : median {np.median(d):.1f}s  mean {d.mean():.1f}s  max {d.max():.1f}s")
    if len(all_iat)>2:
        iat=np.array(all_iat); cv=iat.std()/iat.mean()
        print(f"  inter-arrival time: median {np.median(iat):.1f}s  mean {iat.mean():.1f}s  CV {cv:.2f}")
        patt=("REGULAR/periodic (CV<0.75)" if cv<0.75 else
              "RANDOM/Poisson-like (CV~1)" if cv<1.25 else "BURSTY/CLUSTERED (CV>1.25)")
        print(f"  arrival pattern   : {patt}")
    return tot_ep

# ---- ACF of the fragmentation indicator (persistence) ----
def acf_persistence():
    rows=load_rows([sorted(glob.glob(TEST))[0]])
    x=np.array([1.0 if r.get("regime")==FRAG else 0.0 for r in rows]); x=x-x.mean()
    ac=np.correlate(x,x,"full")[len(x)-1:]; ac/=ac[0]
    # first lag where ACF drops below 1/e
    below=np.where(ac<1/np.e)[0]
    tau=int(below[0]) if below.size else len(ac)
    print(f"\n  frag-indicator autocorrelation: decays below 1/e at ~{tau} ticks (~{tau*0.5:.0f}s)")
    return tau

# ---- honest significance of FRAG-ahead AUC ----
def significance(H, block=None):
    Xtr,ytr,_ = build(TRAIN,H)
    Xte,yte,segs = build(TEST,H)
    clf=HistGradientBoostingClassifier(random_state=42,class_weight="balanced",
        learning_rate=0.08,max_iter=400,max_leaf_nodes=31,l2_regularization=1.0,
        early_stopping=False).fit(Xtr, ytr.astype(int))
    p=clf.predict_proba(Xte)[:,1]
    obs=roc_auc_score(yte,p)
    block=block or 200                     # >> autocorrelation length, so blocks ~ independent
    # naive tick bootstrap (WRONG: assumes independent ticks)
    naive=[]
    for _ in range(400):
        s=rng.integers(0,len(yte),len(yte))
        if yte[s].min()!=yte[s].max(): naive.append(roc_auc_score(yte[s],p[s]))
    # moving-block bootstrap within each session (HONEST: preserves autocorrelation)
    blk=[]
    for _ in range(400):
        yb=[]; pb=[]
        for a,b in segs:
            L=b-a; ys=yte[a:b]; ps=p[a:b]
            nb=max(1,L//block)
            for _k in range(nb):
                st=rng.integers(0,max(1,L-block)); yb.append(ys[st:st+block]); pb.append(ps[st:st+block])
        yb=np.concatenate(yb); pb=np.concatenate(pb)
        if yb.min()!=yb.max(): blk.append(roc_auc_score(yb,pb))
    # rotation permutation null (HONEST: preserves label autocorrelation, breaks assoc)
    null=[]
    for _ in range(400):
        yr=np.empty_like(yte)
        for a,b in segs:
            L=b-a; sh=rng.integers(1,L); yr[a:b]=np.roll(yte[a:b],sh)
        if yr.min()!=yr.max(): null.append(roc_auc_score(yr,p))
    null=np.array(null); pval=(np.sum(null>=obs)+1)/(len(null)+1)
    def ci(a): a=np.array(a); return (np.percentile(a,2.5),np.percentile(a,97.5))
    print(f"\n  H={H:>2}s  observed FRAG-ahead AUC = {obs:.3f}   (test frag-ahead positives={int(yte.sum())})")
    print(f"        naive tick-bootstrap 95% CI : [{ci(naive)[0]:.3f}, {ci(naive)[1]:.3f}]  <- too tight (wrong)")
    print(f"        BLOCK-bootstrap  95% CI     : [{ci(blk)[0]:.3f}, {ci(blk)[1]:.3f}]  <- honest")
    print(f"        rotation null AUC mean {null.mean():.3f} (max {null.max():.3f});  p-value = {pval:.4f}")
    return obs, ci(blk), pval

print(); ep=episode_stats(); tau=acf_persistence()
print("\n"+"="*78); print("FRAG-AHEAD FORECAST AUC -- episode-aware significance"); print("="*78)
for H in (5,15,30): significance(H)
print("\nDONE.")
