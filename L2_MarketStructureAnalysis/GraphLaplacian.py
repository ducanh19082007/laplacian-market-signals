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

#