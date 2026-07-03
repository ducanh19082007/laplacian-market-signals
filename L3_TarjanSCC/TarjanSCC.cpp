// L3 — Tarjan SCC + per-component Bellman-Ford, in C++ 
//
// Why this lives in C++ and not Python: the per-SCC negative-cycle search is the
// O(V*E) hot path of the whole pipeline (L1's find_arbitrage). Tarjan itself is
// only O(V+E) -- cheap -- but it and the Bellman-Ford sweep it feeds want to be
// on the SAME side of the FFI boundary, so we do BOTH here and hand Python back
// finished cycles. Python never loops over SCCs; that loop is the `for (scc...)`
// below.
//
// Contract (see TarjanSCC.py for the Python half):
//   IN : num_nodes, edges = [(u, v, weight), ...]   weight is the -ln(rate) L1
//        already computed in log_transform(). Node ids are 0..num_nodes-1.
//   OUT: list of cycles, each a CLOSED node-id path [start, ..., start] whose
//        rates multiply to > 1 (i.e. sum of weights < 0). At most one cycle per
//        strongly-connected component, matching L1's one-cycle find_arbitrage.
//
// Build: see setup.py  ->  python setup.py build_ext --inplace

#include <pybind11/pybind11.h> //dont worry about this :)
#include <pybind11/stl.h>

#include <vector>
#include <tuple>
#include <algorithm>

namespace py = pybind11;

namespace {

struct Edge {
    int to;
    double weight;
};

// Iterative Tarjan. Recursion would be cleaner but a deep chain of nodes could
// blow the C++ stack, this might be also based on the current Operating System
// one uses like LINUX, anyways, iterative is more efficient to tackle that
// problem; an explicit work stack keeps this safe for any graph size.
// Returns scc id per node, and fills `num_sccs`.

//hence i use iterative just in case the storage got huge...
//for Tarjan SCC, i think this video is the best to demonstrate this:
// https://www.youtube.com/watch?v=wUgWX0nc4NY

//and if someone like me who has not took Data Structure and Algorithms
// aka: EECS 281 at UMich for me at this time (i havent took it at the time i wrote this)
// and did not know what depth first search (DFS) before this, a good reference is:
// https://www.youtube.com/watch?v=7fujbpJ0LB4 

// PERSONAL NOTE: also, i only learned the recursion idea from multiple sources and certain videos and
// the idea of making iteration in the sense of recursion is particular new to me
// bcz i understand that recursion to such a big graph could be a nuisience for operating systems
// i have needed the help from Claude for the idea of creating a struct Frame, each
// frame works. Some one the explaination are written in my voice but explain by either
// Claude or Gemini, i thought i shall provide credits to those LLMs :), Tarjan SCC is very long
// and im frustrated though...

//i wanna do a quick how to TarjanSCC but this will take time so wait for the paper i wanna write...
std::vector<int> tarjan_scc(int n,
                            const std::vector<std::vector<Edge>>& adj,
                            int& num_sccs) {
    const int UNVISITED = -1;
    std::vector<int> index(n, UNVISITED);   // DFS discovery order, -1 = unvisited
    std::vector<int> lowlink(n, 0);
    std::vector<char> on_stack(n, 0);
    std::vector<int> comp(n, UNVISITED);    // result: scc id per node
    std::vector<int> scc_stack;      // Tarjan's component stack

    int next_index = 0;
    int next_scc = 0;

    // Explicit DFS frame: which node, and how far through its adjacency we are.
    struct Frame { int node; size_t edge_i; };
    std::vector<Frame> call;

    for (int root = 0; root < n; ++root) {
        if (index[root] != UNVISITED) continue;
        call.push_back({root, 0});

        while (!call.empty()) {
            Frame& f = call.back();
            int v = f.node;

            if (f.edge_i == 0) {            // first visit to v
                index[v] = lowlink[v] = next_index++;
                scc_stack.push_back(v);
                on_stack[v] = 1;
            }

            bool recursed = false;
            while (f.edge_i < adj[v].size()) { //Process every outgoing edge of v.
                int w = adj[v][f.edge_i].to; //find the trajectory/direction of that edge (to)
                f.edge_i++; // Record that this edge has now been processed.
                // If we later return to this frame, DFS continues
                // from the next edge instead of revisiting this one.
                if (index[w] == UNVISITED) {
                    // Simulate dfs(w);  Pause the current frame and continue with w.
                    call.push_back({w, 0});  // "recurse" into w, add the recursion Frame here
                    recursed = true;
                    break;
                } else if (on_stack[w]) { // "when we got back to a ndoe in a stack
                    // Found a back edge to another node still in the current DFS.
                    // This means v can reach an earlier ancestor, so update
                    // its lowlink accordingly.

                    //put in simple terms, this is only used for when we found the (to) node
                    // is in the stack, and then we changed the lowlink immediately for that,
                    // the backtracking only occurs when all of the node that we can possibly visited are visited!
                    lowlink[v] = std::min(lowlink[v], index[w]);
                }
            }
            if (recursed) continue; // A child DFS was started, so process that child first.

            // Done with v's edges: it's an SCC root iff lowlink == index.
            //this block then will extract one complete SCC once Tarjan algo has determined
            /// that v is the root of SCC. and change the comp as the output.
            if (lowlink[v] == index[v]) {
                while (true) {
                    int u = scc_stack.back();
                    scc_stack.pop_back(); //gradually delete the stacks, both in scc_stack and on_stack both
                    on_stack[u] = 0;    
                    comp[u] = next_scc; //add the realted scc_id into comp to signify the SCC the node is in
                    if (u == v) break; // when reached the loop/final node then stop this 
                }
                next_scc++; //after done notating ids, we give a good start for the next scc
            }
            
            // Simulate returning from dfs(v).
            call.pop_back();

            // After returning, propagate v's lowlink to its parent,
            // exactly like the recursive algorithm does after dfs(child).
            //this is where the backtracking happened!
            if (!call.empty()) {
                int parent = call.back().node; // the previous node for backtracking is in .back() the back
                lowlink[parent] = std::min(lowlink[parent], lowlink[v]);
            }
        }
    }

    num_sccs = next_scc;
    return comp;
}

// Bellman-Ford restricted to one SCC's internal edges. `members` are the node
// ids in this component. Relaxes |members| times; if still relaxing, a negative
// cycle exists -- we walk predecessors back |members| steps to land ON it, then
// collect the closed loop. Returns empty if the component is arb-free.



//Bellman-Ford over the log-weights. Returns one node cycle whose rates
//multiply to > 1 (an arbitrage loop), or None if the market is arb-free.

//If an arbitrage path exists, it returns a readable execution route like:
//[("btc", "Binance"), ("eth", "Binance"), ("eth", "Kraken"), ("btc", "Binance")]

// Call log_transform() first.

// highly recommend this video: https://www.youtube.com/watch?v=B5PmlJACZ9Y  for Bellman-Ford comprehension 

//Bellman-Ford is actually more comprehensible to explain now:
// we can set a certain vertex of s as the cost[s] = 0; (distance of the vertex not edge)
// as s meaning being the source node, and for other verticies as v node and
// cost[v] = inf, you might wanna supplant cost as distance (from v to source s, thats why source cost s = 0)
// while the previous[v] = None for all vertex (for which recognize as -1 -> None)
std::vector<int> bellman_ford_cycle(const std::vector<int>& members,
                                    const std::vector<int>& comp,
                                    const std::vector<std::vector<Edge>>& adj) {
    const double EPS = 1e-12; //error constant, in case sth goes wrong given 
    // we are evaluating exchange rates, some exchange rate are very small and hence can be
    // for go by not putting an error constant (idk if other example of Bellman-Ford does 
    // this or no but for me i think it matters)
    int scc_id = comp[members[0]];
    int m = static_cast<int>(members.size());

    // dist=0 for every node detects ANY negative cycle in the component; within
    // an SCC every node reaches every other, so this 0-init is sound. Index the
    // dist/pred arrays by global node id, sized to the largest id in the SCC.
    int max_id = 0;
    for (int u : members) max_id = std::max(max_id, u);
    std::vector<double> dist(max_id + 1, 0.0);
    std::vector<int> pred(max_id + 1, -1);

    // for the next crazy 3 for loops happening here, i would like to refer to that particular
    // youtube video that i had at the initial comments. 
    int updated = -1;
    for (int iter = 0; iter < m; ++iter) {
        updated = -1;
        for (int u : members) {
            for (const Edge& e : adj[u]) {
                if (comp[e.to] != scc_id) continue;       // stay inside the SCC
                if (dist[u] + e.weight < dist[e.to] - EPS) {
                    dist[e.to] = dist[u] + e.weight;
                    pred[e.to] = u;
                    updated = e.to;
                }
            }
        }
        if (updated == -1) return {};   // converged: no negative cycle here :((
    }

    // `updated` is on or downstream of the cycle; m predecessor hops land on it.
    int node = updated;
    for (int i = 0; i < m; ++i) node = pred[node];

    int start = node;
    std::vector<int> cycle = {start};
    int cur = pred[start];
    // this helps finding the negative cycle
    while (cur != -1 && cur != start) {
        cycle.push_back(cur);
        cur = pred[cur];
    }
    cycle.push_back(start);
    std::reverse(cycle.begin(), cycle.end());
    return cycle;
}

} // namespace

// One closed cycle per non-trivial SCC. Singletons (size 1, no self-loop in this
// graph) can't hold a cycle, so they're skipped -- that's the pruning win.
std::vector<std::vector<int>> find_all_arbitrage(
        int num_nodes,
        const std::vector<std::tuple<int, int, double>>& edges) {

    //initializes all of the adjcent edges, and each verticies have each adjcent edges
    // and that fact will be represented here, the basic framework of a graph in c++
    std::vector<std::vector<Edge>> adj(num_nodes);
    for (const auto& [u, v, w] : edges) {
        adj[u].push_back({v, w});
    }

    int num_sccs = 0;
    std::vector<int> comp = tarjan_scc(num_nodes, adj, num_sccs);
    // from tarjan_scc function, we got ourself a id of all the SCCs that we need
    // and the number of the available SCCs, reason we have this instead of counting is to
    // reduce time complexity as i wanna keep it O(|V| + |E|), and .max_element is an iterator
    // which increase storage and also time to do so. and its simple so...

    // Bucket node ids by component.
    std::vector<std::vector<int>> members(num_sccs);
    for (int v = 0; v < num_nodes; ++v) members[comp[v]].push_back(v); // members of all
    //this consists the index of all the SCCs available in the graph.

    std::vector<std::vector<int>> cycles;
    for (const auto& mem : members) {
        if (mem.size() < 2) continue;   // singleton => no cycle possible
        std::vector<int> cyc = bellman_ford_cycle(mem, comp, adj);
        if (!cyc.empty()) cycles.push_back(std::move(cyc)); 
        // we want to transfer it right away to optimize everything (im pretti greedi u know?)
    }
    //finallly this is the steps that goes through every SCCs 
    // found and then perform ballman_ford_adj
    return cycles;
}

// for this part, we decided the "custom library of TarjanSCC + Bellman-Ford algorithms on negative
// cycles will be named tarjan_arb for that one should initialize this with this command
// cd L3_TarjanSCC && python setup.py build_ext --inplace

//the rest should look at setup.py, more explaination will be given in the paper.
PYBIND11_MODULE(tarjan_arb, m) {
    m.doc() = "L3: Tarjan SCC + per-component Bellman-Ford loop arbitrage search";
    m.def("find_all_arbitrage", &find_all_arbitrage,
          py::arg("num_nodes"), py::arg("edges"),
          "Return one closed arbitrage cycle (node-id path) per SCC that holds one.");
}

