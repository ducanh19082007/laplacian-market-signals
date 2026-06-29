# In L1 subgraph() vs Tarjan SCC — two halves of the same step
#They aren't competing things; they're producer and consumer. 
# Tarjan decides which nodes; subgraph() builds the smaller 
# graph from that decision.

#Tarjan SCC (L3, not yet written): A graph algorithm 
# that partitions all nodes into strongly-connected components — 
# maximal groups where every node can reach every other node by 
# following edge directions. The key fact it exploits: a 
# cycle must live entirely inside one SCC. So any arbitrage loop 
# (which is a cycle) can never span two SCCs. It also throws away 
# singletons (SCCs of size 1 — nodes that aren't part of any cycle). 
# 
# Output: a list of node-sets, one per non-trivial component. 
# It computes node-sets; it does not touch the graph itself.

#subgraph(nodes) (L1, already written): Pure mechanical reduction. 
# Given a node-set, it returns a new ExchangeRateGraph holding only 
# those nodes and the edges with both endpoints inside. No analysis, no decision — 
# just "carve out this slice." It shares edge dicts by reference, 
# so the -ln(rate) weights aren't recomputed.

#So the round trip is:


#Tarjan(g)  →  [ {USD,EUR,GBP}, {JPY,AUD}, ... ]     # L3 decides WHICH nodes
#g.subgraph({USD,EUR,GBP})  →  small graph           # L1 BUILDS the slice
#small_graph.find_arbitrage()  →  cycle              # L1 Bellman-Ford on the slice

#The difference in one line: Tarjan is the brain (which nodes can possibly contain a cycle), subgraph is the hands (assemble that slice so Bellman-Ford runs over |tiny| edges instead of |all| edges). The payoff is turning one O(V·E) sweep over the whole graph into a sum of much cheaper sweeps over tiny components. Right now NullSpatial returns None, so this whole optimization is bypassed and the engine just searches the full graph once.