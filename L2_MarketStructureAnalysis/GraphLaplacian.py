#FROM THE IDEA OF USING SPECTRAL GRAPH THEORY

#L = D - A, AND THEN FIND THE SPECTRAL GAP, TROPICAL EIGNEVALUE, STRAIN DEVIATION, and a bit of Tropic Mathematics

# theortically, we just need to use the Bellman-Ford algo then it is done, but then 
# with spectral graph theory, we can give deep down which part of the graph 4
# is unstable and possibly have a negative cycle in which is signaled with that in their
# list of eigenvalues, there exist a negative real part of an eigenvalues. 
# The presence of a negative eigenvalue is a mathematical proof that a 
# negative cycle exists somewhere in the network. 
# The magnitude of the negative eigenvalue tells you the "strength" 
# or severity of the market instability (wider arbitrage margins).
# doing so reduces the run time of the algorithms and also finds the 
# spectral graph which shows
# how healthy the market currently is as a way to predict and forecast the gap

#GOAL:
#λ₂ + strain → Market Health dashboard (the snapshot view)
#tropical eigenvalue → go/no-go gate for arbitrage detection
#strain history → L5's OU/ADF regime engine

L2 is a photo of the pond right now. It looks at the whole surface and tells you two things: is the water all connected (so a ripple anywhere spreads everywhere), or has it split into separate puddles that don't talk to each other? And: is the surface flat, or is it currently disturbed — bumpy, out of its resting state? That "how disturbed is it right now" number is strain. A photo is instant. It's a great description of this moment, but a photo can't tell you whether a bump is about to settle down or about to get worse.

L5 watches the video of the pond over time. It doesn't care about the whole surface — it just watches that one "how disturbed" number bounce around tick after tick. By watching the history, it learns the pond's personality: when this pond gets disturbed, does it calm back down quickly, slowly, or does the disturbance just keep growing? That's the part you can't get from a single photo — you only learn it by watching how the bumps behave over time.

How they help each other:

L2 feeds L5 its raw material. L5 has nothing to watch unless L2 keeps handing it that "how disturbed right now" number, tick after tick. No L2 strain → L5 is blind.

L5 gives L2's snapshot meaning. On its own, L2 saying "the pond is bumpy right now" doesn't tell you whether to care. L5 is the one that says "this pond always calms down fast, so that bump is an opportunity that's about to close" versus "this pond's bumps tend to keep growing, so don't trust it." L2 spots the disturbance; L5 tells you whether it's the kind that bounces back.


And the real payoff is when they disagree. L2's photo might say "the pond looks healthy and well-connected" while L5's video says "but this particular spot hasn't actually been calming down lately." That mismatch — the snapshot looks fine but the behavior over time says otherwise — is the most useful warning the whole system can give you. Neither one alone would catch it.

So: L2 = what's happening right now. L5 = what that usually leads to. One describes, the other predicts, and they're checking the same thing (how well the market fixes itself) from two completely different angles.

│               LAYER 2 MATRIX ANALYSIS                  │
│  • Calculate λ_trop  ──► Identifies the true target μ  │
│  • Calculate λ_2     ──► Identifies the speed θ        │
└───────────────────────────┬────────────────────────────┘
                            │ (Real-Time Parameter Injection)
                            ▼
┌────────────────────────────────────────────────────────┐
│            LAYER 3 ORNSTEIN-UHLENBECK SDE             │
│   dX_t = θ_λ2 * (μ_λtrop - X_t) dt + σ_noise * dW_t    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                MONTE CARLO SIMULATION                  │
│ Run 10,000 paths at horizon Δt to see if loop survives │
└────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd

from L1_DataProcessing.DataProcessing import ExchangeRateGraph

graph = ExchangeRateGraph(assets, transfer_cost = 0.1, fee = 0.1, quote_window=0.2)