import json, glob, math, os, sys

# Resolve data relative to the REPO ROOT (parent of TestCases/), not the current
# working directory -- so this runs the same whether launched from the repo root
# (`python TestCases/TestStressRegime.py`) or from inside TestCases/ (the IDE's
# default cwd). Using a cwd-relative glob was why it found 0 rows and crashed.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows=[]
for p in sorted(glob.glob(os.path.join(_ROOT, "data_examples", "*.jsonl"))):
    for ln in open(p):
        ln=ln.strip()
        if not ln: continue
        try: r=json.loads(ln)
        except: continue
        if "ts" in r: rows.append(r)
rows.sort(key=lambda r:r["ts"])
S=[r for r in rows if r["regime"]=="STRESSED"]

if not rows:
    sys.exit(f"no data rows found under {os.path.join(_ROOT, 'data_examples')} "
             f"-- generate some with the L4 engine first.")
if not S:
    sys.exit(f"loaded {len(rows)} rows but none are STRESSED -- nothing to check.")

print("=== 1) MATH CONSISTENCY: does lam == ln(net_return)/hops ? ===")
for r in S:
    c=r["cycles"][0]
    hops=len(c["path"])-1
    implied=math.log(c["ret"])/hops
    print(f"  t={r['t']:6.1f}  hops={hops}  ln(ret)/hops={implied:.4f}  "
          f"logged lam={r['lam']:.4f}  diff={abs(implied-r['lam']):.4f}")

print("\n=== 2) STRUCTURAL DEPENDENCY: transfer (cross-venue) vs convert hops ===")
for r in S:
    c=r["cycles"][0]
    path=c["path"]
    transfers=converts=0
    for u,w in zip(path,path[1:]):
        au,vu=u.split("@"); aw,vw=w.split("@")
        if au==aw and vu!=vw: transfers+=1
        else: converts+=1
    print(f"  t={r['t']:6.1f}  {transfers} cross-venue TRANSFER hops + {converts} convert hops   (+{c['profit_pct']:.2f}%)")

print("\n=== 3) THE GAP: top EFFICIENT lam_raw vs STRESSED lam_raw ===")
eff=sorted((r.get('lam_raw') or -9) for r in rows if r['regime']=='EFFICIENT')
print("  highest 8 EFFICIENT lam_raw:", [f'{x:.4f}' for x in eff[-8:]])
print("  all 6 STRESSED  lam_raw:", sorted(round(r["lam_raw"],4) for r in S))
print(f"  ratio (min STRESSED / max EFFICIENT) = {min(r['lam_raw'] for r in S)/eff[-1]:.1f}x")

print("\n=== 4) live threshold check: STRESSED iff net lam > 5*tau ===")
tau=-math.log(1-0.00023)
print(f"  tau={tau:.6f}  stress line=5*tau={5*tau:.6f} (={5*tau*100:.4f}%/hop)")
print(f"  smallest STRESSED net lam = {min(r['lam'] for r in S):.4f}  -> {min(r['lam'] for r in S)/(5*tau):.0f}x over the line")
