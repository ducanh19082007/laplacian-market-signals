# laplacian-market-signals

A real-time, cross-venue market structure engine for cryptocurrency exchange rates. It builds a live directed graph of currency pairs across multiple brokers, analyses the graph's structure and dynamics, and turns that analysis into three concrete outputs: a **market health signal**, a **per-pair regime classification**, **and ML feature engineering for trading signals**.

> Done Implementing the Analysis, Live Graph, Multi Broker Order Book, Currently in ML development

Disclaimer: This repo also has Claude as a big help. Hence if there exists certain errors or unreadable code, such as URLmethods.py, then excuse me.

Inspiration:

[1] D. A. Spielman, "Spectral and Algebraic Graph Theory," Yale University.
    Available: http://cs-www.cs.yale.edu/homes/spielman/sagt/sagt.pdf (*)

[2] I. Akouaouch and A. Bouayad,
    "An Innovative Approach to Identifying Triangular Arbitrage Opportunities
    in Financial Markets Using the Bellman–Ford Algorithm,"
    Bulletin of Electrical Engineering and Informatics,
    vol. 14, no. 3, pp. 2035–2047, 2025.
    Available: https://beei.org/index.php/EEI/article/view/10817

[3] B. A. Mason,
    "Tropical Algebra, Graph Theory, and Foreign Exchange,"
    Honors Thesis,
    James Madison University, 2020.
    Available: https://commons.lib.jmu.edu/honors201019/304

[4] R. E. Tarjan,
    "Depth-First Search and Linear Graph Algorithms,"
    SIAM Journal on Computing,
    vol. 1, no. 2, pp. 146–160, Jun. 1972.
    doi: 10.1137/0201010.

[5] S. Mallik,
    "Pricing Cryptocurrencies: Modelling the ETHBTC Spot-Quotient Variation
    as a Diffusion Process,"
    arXiv:2111.11609 [q-fin.ST], Nov. 2021.
    Available: https://arxiv.org/abs/2111.11609

[6] M. Fiedler,
    "Algebraic Connectivity of Graphs,"
    Czechoslovak Mathematical Journal,
    vol. 23, no. 2, pp. 298–305, 1973.
    doi: 10.21136/CMJ.1973.101168.

[7] U. von Luxburg,
    "A Tutorial on Spectral Clustering,"
    arXiv:0711.0189 [cs.DS], Nov. 2007.
    Available: https://arxiv.org/abs/0711.0189

[8] J. Bang-Jensen,
    "Finding Negative Cycles,"
    Department of Mathematics and Computer Science,
    University of Southern Denmark.
    Available: https://www.imada.sdu.dk/u/jbj/DM817/Negativicyclefinding.pdf :contentReference[oaicite:0]{index=0}

---
# In development, ML Applications

ML Model planned to added for efficient detections:
1. Anomaly detection on market state ( isolation forest, an autoencoder's reconstruction error, or even a rolling Mahalanobis distance)
2. Regime / fragmentation forecasting
3. "Does this dislocation close?"
4. Structure → forward-volatility study (a whole study and check so no need to see good stuffs)

suggestion on data engineering

The classic sample-size heuristic for a prediction model is EPV: ~10–20 events (minority-class instances) per feature (Peduzzi et al. 1996; tightened by Riley et al. 2019). If you use ~8 features (λ, λ₂, connectivity, strain, n_components + a couple of lags), you want on the order of 80–160 STRESSED/FRAGMENTING examples minimum just to not overfit

The gold-standard, no-guessing method: gather a pilot, then train your model on 25% / 50% / 75% / 100% of it and plot validation score vs training size.

Curve still rising steeply at 100% → you need more data.
Curve flattened → you have enough.

ticks are 0.5s apart and highly autocorrelated — λ₂ is persistent, that's the entire premise of the project. So 172,800 rows/day are nowhere near 172,800 independent samples. A STRESSED episode lasting 20s is 40 rows but ≈ one independent event.

The deflation is real and estimable. For an AR(1)-ish series with autocorrelation ρ:
$$n_{\text{eff}} \approx n \cdot \frac{1-\rho}{1+\rho}$$
or more practically, n_eff ≈ (total seconds) / (decorrelation time τ). If regimes persist for ~minutes, a full day gives you maybe hundreds of independent episodes, not 173k. This is exactly why "just log a day, that's 173k rows, plenty!" is a trap — your effective n is 100–1000×smaller. Counting episodes (contiguous regime runs) instead of rows corrects for this automatically.


| Wall-clock | Rows | Good for |
|---|---|---|
| 5 hours | ~36k | a taste; not enough rare events |
| 2–3 days | ~350–520k | anomaly detection (#1) + structure study (#4)|
| 1–2 weeks | ~1.2–2.4M | first real go at regime forecasting (#2) |
| 1–3 months | 5M+ | robust #2 / cycle-closing (#3), ideally spanning a volatile event (a crash/rally) |

+ drop the first ~2 min of each session and any row with n_nodes below ~60.
+ Stick to stress_raw_within_h for study #4.4
+ Keep fiedler + n_components; drop connectivity and strain
+ lam_raw has real, live signal — median 6.1e-5, p90 1.8e-4, spikes to 4.5e-2. That spread + occasional spikes is exactly what you want to predict. Your training signal is alive.
 
ALSO A LEARNING CURVE MIGHT BE A GOOD IDEA

The training error increases as you increase the size of your dataset, because it becomes harder to fit a model that accounts for the increasing complexity/variability of your training set.

The test error decreases as you increase the size of your dataset, because the model is able to generalise better from a higher amount of information.

All the data will then sent and got trained in drive and by google colab, the data in the data folder is mainly for show, with .jsonl as the actual data, and meta.json is values that got used in the corressponding data

Certain development are developed currently in Google Colab, if interested, you can send me access request:

Anomaly Detection (unfinished): https://colab.research.google.com/drive/142Bw_B8CnhuNmOaVWGXIkuhPeRtyqpN6#scrollTo=nKwsRpCXzjGP

For the Test cases, i already test those stuffs outside this repo, but i wanted to double check so they will be done in the near future
---

## How it works

Every tick, live order book data from multiple exchanges is assembled into a single directed graph where nodes are `(asset, venue)` pairs and edges are exchange rates — both within a venue (trading) and across venues (transferring). That graph is analysed two ways: structurally, via spectral graph theory, and dynamically, via a calibrated stochastic process fit to its history. Triangular and cross-venue arbitrage cycles are detected as a side effect of this analysis, not as the end goal.

The point of building it this way: the same underlying question — *how efficiently does this market correct itself when something pushes it out of equilibrium* — turns out to answer three different practical questions depending on how you frame it. That reframing is the actual subject of this README.

![alt text](Graph.jpg)

Interpretation:

- a live arbitrage-cycle scanner with realistic cost gating

- a live market-structure/regime monitor.

Signals:

- Tropical (max‑plus) eigenvalue λ: The maximum cycle mean of the ln(rate) weights = the best per‑hop log‑return any loop in the market offers. exp(λ) is the geometric-mean rate per hop. and can be interperted to be how mispriced the market is right now, Nonetheless, at 1 Hz this has ~zero autocorrelation and reverts faster than round-trip latency, so it is descriptive, not predictive. Don't read a spike as "go trade" — read it as "the market is dislocated this instant."

- Fiedler value λ2: High → tightly coupled market; mispricing propagates and closes fast (healthy, well-arbitraged).
Near 0 → the graph is barely holding together; deviations can persist. It's deliberately unweighted/from affinity/similarity matrix so that zero-cost same-asset transfer edges (weight −ln 1 = 0) don't spuriously split the market by venue.

---

## The Multi-Venue Graph Model

Nodes are `(asset, venue)` pairs, not just assets — `BTC@Binance` and `BTC@Coinbase` are distinct nodes. Two edge types exist: intra-venue trading edges (`Wᵢⱼ = -ln(bid or 1/ask)`) and inter-venue transfer edges (`Wᵢⱼ = -ln(1 - transfer_fee)`). A profitable loop across both edge types — say, sell ETH for BTC on Binance, transfer BTC to Coinbase, sell BTC for ETH, transfer back — only beats the combined trading and transfer fees if the price gap between venues is large enough. Cross-venue cycles are also gated by transfer latency: on-chain settlement can take seconds to minutes, by which point the gap that created the opportunity is often gone, which is why pre-funded balances on every venue (collapsing the transfer leg to zero-latency internal accounting) are the practical way this would ever be exploitable.

---

## Initializing

\ python L3_TarjanSCC/TarjanSCC.py 
\ python -m L3_TarjanSCC.TarjanSCC
\ from L3_TarjanSCC.TarjanSCC import find_all_arbitrage
\ cd L3_TarjanSCC && python setup.py build_ext --inplace

---

## Architecture

### Layer 1 — Data Preprocessing
Order books from every connected venue are merged into one unified state keyed by `(asset, venue)`. Intra-venue and inter-venue edges are constructed each tick and log-transformed into weight matrix `W`, turning the multiplicative arbitrage problem into an additive one: a profitable loop becomes a negative-weight cycle. ![alt text](image-2.png)

Furthermore, this stage also help showing how normal arbitrage loop without the further layers can be like. The implementation of L2 -> L4 is mainly for pushing the graph finding quicker and more efficient and also to conclude different aspects of the market such as L2 can shows the "market health", if spectral sum goes up, then its healthy and vice versa. Appearantly, we want it to go down, then when it goes down, there is more volatility then the arbitrage occurs better

### Layer 2 — Market Structure Analysis
Three structural signals come from the Graph Laplacian `L = D - A`. The spectral gap `λ₂` (second-smallest Laplacian eigenvalue) measures how well-connected the market is right now — high means mispricing propagates and closes quickly, low means the graph is fragmented and deviations may persist. The tropical (min-plus) eigenvalue is the **minimum cycle mean** of the `-ln(rate)` weights: it is negative exactly when a profitable cycle exists anywhere in the graph, so it acts as a cheap go/no-go gate — when it is `≥ 0` (no profitable cycle possible) the reduction and detection are skipped entirely for that tick. Strain measures how far the observed graph deviates from the no-arbitrage equilibrium where every cycle's product equals 1. This layer is diagnosis only — it scores the whole market and gates; it does **not** shrink the graph (that is Layer 3).

### Layer 3 — Spatial Graph Analysis
This is the layer that actually makes the graph smaller. Tarjan's algorithm finds Strongly Connected Components and drops the singletons — a profitable cycle must live *entirely* inside one SCC, so the subsets where no closed trading loop is possible (illiquid pairs, disconnected venues) fall out before Bellman-Ford ever runs. Spectral embedding then *ranks* the remaining SCCs by instability so the most promising one is searched first — but ranking only reorders, it never drops nodes (a spectral cut could slice through a real cycle and lose the arbitrage). Layer 3's output is the node-set of each non-trivial SCC. 
![alt text](image-1.png)

### Arbitrage Detection — the L3 → L1 round trip (no separate layer)
There is no standalone detection layer. Detection is the loop closing back on Layer 1: for each SCC node-set Layer 3 marks tradeable, the engine calls `ExchangeRateGraph.subgraph(scc)` to shrink the adjacency, then runs that smaller graph's own `find_arbitrage()` — the Bellman-Ford that already lives in L1's DataProcessing. This turns the `O(V·E)` sweep over the whole graph into a sum over tiny components. Detection runs only on cycles that Layer 5 confirms are in a statistically mean-reverting regime — a cycle in a trending or random-walk regime might widen instead of close, so a negative cycle there is not the same thing as real arbitrage. Surviving cycles pass a cost filter (spread, depth, transfer fees) and are written to the Feature Store with their full path and profit. Until Layer 3 is real, the spatial stub returns `None` and the engine searches the whole graph once.

### Layer 5 — Regime & Risk Engine
This layer no longer tries to *forecast* arbitrage. An earlier version modelled the tropical eigenvalue as an Ornstein-Uhlenbeck process and forecast whether the next tick would clear the fee — but on a 1-second tape that eigenvalue has ~zero one-step autocorrelation, and any opening reverts in well under the round-trip execution latency, so a *tradeable* forecast is not possible at this observation scale. That is the market being efficient, not the model failing. (The OU work is preserved on the `archive/ou-arbitrage-attempt` branch.)

So Layer 5 **measures** the market's regime rather than predicting it, from two spectra of the live graph every tick:
- **arbitrage intensity** — the tropical (max-plus) eigenvalue `λ`: the best per-hop cycle return, i.e. how mispriced the market is right now;
- **connectivity** — the Fiedler value `λ₂` (algebraic connectivity of the *unweighted* Laplacian, so zero-cost cross-venue transfer edges don't spuriously split the graph) plus the number of connected components. Unlike the single-tick arb spikes, connectivity is *persistent*, so it is actually usable at 1 Hz.

Each tick is classified into one of three regimes:
- **EFFICIENT** — connected and quiet: `λ` at/under the fee, prices consistent across venues (the normal, well-arbitraged state);
- **STRESSED** — still one connected market, but `λ` far above the fee: large dislocations are open (a fast move, one venue lagging, a volatility spike) while the graph is structurally intact;
- **FRAGMENTING** — the graph has split into ≥ 2 components (or `λ₂ ≈ 0`): venues/assets decoupling, liquidity withdrawing. The top structural risk, so it overrides regardless of `λ`.

to put it simply in arbitrage trading:
- EFFICIENT → gaps too small, already competed away → noise
- STRESSED → gap is real AND closes → trade
- FRAGMENTING → gap is real but does NOT close (or can't execute) → trap

`regime_engine.py` streams this live — a terminal readout plus a rolling 2-D regime map of connectivity vs arbitrage-intensity — and ships an offline `--demo` that classifies all three regimes with no feed. ADF/Monte-Carlo reversion features remain a possible future add-on, but only for connectivity, which has the persistence the arb intensity lacks.

![alt text](image.png)


![alt text](image-3.png)

# just watch, save nothing:
.venv/bin/python "L4_Regime&RiskEngine/regime_engine.py" --no-store

# headless, no file either:
.venv/bin/python "L4_Regime&RiskEngine/regime_engine.py" --headless --no-store

### Layer 6 — Execution Engine *(planned)*
Async order routing and atomic cross-leg execution, built only on signals that have cleared the regime gate and the feature store's validation. Not the current focus.

---

## What this project can actually deliver

Three concrete things come out of this pipeline, and they're not independent products bolted together — they're the same underlying measurement (how efficiently this market self-corrects) read at three different resolutions.

**Market health.** The spectral gap and strain, tracked against a rolling baseline, tell you whether the market is currently well-connected or fragmented — a live diagnostic, alertable when a venue or asset cluster starts diverging from its normal behaviour. This is the snapshot view: is the market okay right now.

**Regime classification.** Calibrating θ per pair and validating it with an ADF test tells you whether that specific pair is currently mean-reverting (efficient, deviations correct) or trending (inefficient, deviations persist or grow). This is the time-series view: what kind of market behaviour is this pair exhibiting, and is a detected arbitrage cycle even likely to close. It's also what separates this project from a naive scanner — a negative cycle only means something once the regime confirms it's the closing kind.

**Trading signals.** Every structural metric, regime classification, and detected cycle gets logged into a timestamped feature store, alongside forward-looking labels (realised return, or probability of reversion within a horizon). That dataset is the input to a predictive model — the spectral and regime metrics stop being descriptive and become engineered features with a testable claim: does market structure predict what happens next.

The interesting part is where these three disagree. If the graph looks structurally healthy but θ shows a pair isn't actually reverting, that mismatch — not either metric alone — is the most informative signal the system produces.

---

## Current status

| Layer | Status |
|---|---|
| WebSocket feed + order book dashboard (Binance) | ✅ Done |
| K4 intra-venue graph (6 live pairs) | ✅ Done |
| Multi-broker aggregator (Binance live, OANDA/IBKR mock) | ✅ Done |
| FX graph + log transform | ✅ Done |
| Bellman-Ford cycle detection (in L1 DataProcessing) | ✅ Done |
| Subgraph reduction + per-SCC detection wiring (L3 → L1 round trip) | ✅ Done |
| Multi-venue graph (asset × venue nodes, transfer edges) | ✅ Done |
| Spectral structure (Laplacian, λ₂, tropical eigenvalue, strain) | ✅ Done |
| Regime engine (spectral EFFICIENT / STRESSED / FRAGMENTING classifier) | ✅ Done |
| Spatial analysis + SCC pruning | ✅ Done |
| OU arbitrage forecasting | ❌ Shelved — no 1 Hz predictability (archived) |
| Regime-gated arbitrage detection | 📋 Planned |
| Feature store + forward labels | 📋 Planned |
| ML model / signal validation | 📋 Planned |
| Execution engine | 📋 Planned |

---

## Stack

- **Python** — core pipeline, asyncio event loop
- **websockets** — multi-venue order book feeds
- **numpy / scipy** — Laplacian construction, eigenvalue computation
---

## Disclaimer

This is a research and feature-engineering tool, not an automated trading system. Detected cycles and regime classifications are logged for analysis, not executed. Live execution at retail scale competes against co-located, microsecond-latency infrastructure that this project has no intention of trying to beat — the value here is in the structural and statistical analysis, and in producing a defensible, testable dataset, not in chasing fleeting price gaps.

---
