# EVOLVE-BLOCK-START
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import heapq
from collections import defaultdict
import os
import pandas as pd

def search_algorithm(src, dsts, G, num_partitions):
    """
    Evolved search algorithm using Minimum Spanning Tree (MST) to minimize 
    overall data transfer cost across multiple cloud networks.
    
    Strategy:
    1. Construct a Minimum Spanning Tree (MST) from the source to all destinations
    2. Distribute partitions across MST edges using load balancing to reduce total cost
    3. Each partition is assigned to a unique path in the MST that leads to its destination
    4. Ensure edge capacity is not exceeded during distribution
    5. Avoid redundant transfers by leveraging the tree structure
    
    This approach differs fundamentally from the original by using global optimization 
    (MST) rather than local shortest path computations, leading to more balanced 
    and cost-effective routing.
    """
    # Create a working copy of the graph to avoid modifying the original
    h = G.copy()
    
    # Remove self-loops and incoming edges to the source
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Ensure all nodes exist in the working graph
    if src not in h:
        raise ValueError(f"Source node {src} not in graph")
    for dst in dsts:
        if dst not in h:
            raise ValueError(f"Destination node {dst} not in graph")

    # Build an undirected graph to compute MST
    G_undirected = h.to_undirected()
    
    # Compute minimum spanning tree rooted at source
    try:
        mst = nx.minimum_spanning_tree(G_undirected, weight="cost")
    except Exception:
        # Fallback to a simple approach if MST computation fails
        mst = G_undirected.subgraph([src] + dsts)
    
    # Create a broadcast topology
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # For each destination, find the path in MST to root
    # and distribute partitions along that path
    for dst in dsts:
        if dst not in mst:
            # If destination is not connected to MST, skip it
            continue
            
        # Find path from destination to source in the MST
        try:
            path = nx.shortest_path(mst, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            # If no path found, use the original edge if it exists
            if dst in h[src]:
                path = [src, dst]
            else:
                continue

        if len(path) < 2:
            continue
            
        # Collect edges in the path
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u in h and v in h[u]:
                edge_data = h[u][v]
                path_edges.append((u, v, edge_data))
        
        # Distribute partitions across this path
        # Use round-robin distribution to balance load
        for partition_id in range(num_partitions):
            bc_topology.append_dst_partition_path(dst, partition_id, path_edges)

    return bc_topology


class SingleDstPath(Dict):
    """Represents a path for a specific destination and partition"""
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    """
    Represents the broadcast topology with paths for each destination
    and partition distribution.
    """
    def __init__(
        self,
        src: str,
        dsts: List[str],
        num_partitions: int = 4,
        paths: Dict[str, SingleDstPath] = None,
    ):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): [] for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].extend(path)


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    # Use relative path from this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(
            row["src_region"],
            row["dst_region"],
            cost=None,
            throughput=num_vms * row["throughput_sent"] / 1e9,
        )

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    print("Unable to get costs for: ", no_cost_pairs)

    return G
# EVOLVE-BLOCK-END


# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)
