import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int


class Node:
    """
    Graph node with auto-incrementing ID for unique identification.
    """

    __slots__ = "name"
    _id = 0

    def __init__(self):
        self.name = Node._id
        Node._id += 1

    def __lt__(self, other) -> bool:
        if not isinstance(other, Node):
            raise ValueError
        return self.name < other.name


class Edge:
    """
    Edge connecting 2 or more nodes (supports hyperedges).
    """

    __slots__ = "nodes"

    def __init__(self, *nodes):
        self.nodes = nodes


class EqxGraph(eqx.Module):
    """
    Stores node-to-index mappings for efficient array operations.

    The dictionaries restrict batching over different graph topologies.
    """

    node_to_global: dict[Node, int]
    node_to_local: dict[Node, tuple[int, int]]
    block_to_global: list[Int[Array, " block_size"]]


class BlockGraph:
    """
    General block graph structure for efficient block-wise operations.

    A graph G = (V, E) where vertices are organized into blocks for
    parallelized computation.
    """

    def __init__(self, blocks: list[list[Node]], edges: list[Edge]):
        # Flatten blocks to create global node ordering
        nodes = [item for sublist in blocks for item in sublist]
        self.node_to_global = {node: ind for ind, node in enumerate(nodes)}
        self.node_to_local = {}
        for i, block in enumerate(blocks):
            # (which block, where in that block)
            self.node_to_local.update(
                {node: (i, ind) for ind, node in enumerate(block)}
            )
        self.block_to_global = [
            jnp.array([self.node_to_global[node] for node in block]) for block in blocks
        ]

        # something to think about, with the array representation generated later
        # we could presumable vmap over graph topologies, but we don't want to be
        # limited by the static fluff
        self.nodes = nodes
        self.edges = edges
        self.blocks = blocks

    def get_sparse_adj_list(
        self,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]], int, list[int]]:
        """
        Generate sparse adjacency lists organized by edge cardinality.
        """
        max_k = max([len(edge.nodes) for edge in self.edges]) - 1
        degrees = [0 for _ in self.nodes]
        neighbors = [[[] for _ in range(max_k)] for _ in range(len(self.nodes))]
        edge_ids = [[[] for _ in range(max_k)] for _ in range(len(self.nodes))]

        for eid, edge in enumerate(self.edges):
            idxs = [self.node_to_global[v] for v in edge.nodes]
            k = len(edge.nodes)
            ind = k - 2
            for pos, u in enumerate(idxs):
                degrees[u] += 1
                others = idxs[:pos] + idxs[pos + 1 :]
                neighbors[u][ind].append(others)
                edge_ids[u][ind].append(eid)

        return neighbors, edge_ids, max_k, degrees

    def get_padded_adj_arr(
        self,
    ) -> tuple[
        list[Int[Array, "num_nodes max_edges max_k-1"]],
        list[Int[Array, "num_nodes max_edges max_k-1"]],
        list[Int[Array, "num_nodes max_edges"]],
    ]:
        """
        Convert sparse adjacency lists to padded dense arrays for efficient JAX ops.
        """
        neighbors, edge_ids, max_k, degrees = self.get_sparse_adj_list()

        padded_adj_arrays = []
        mask_arrays = []
        edge_id_arrays = []

        for block_gidxs in self.block_to_global:
            gidxs = np.asarray(block_gidxs, dtype=np.int32)
            num_nodes = int(gidxs.size)
            max_degree = max(degrees[u] for u in gidxs)

            padded = np.zeros((num_nodes, max_degree, max_k), dtype=np.int32)
            mask = np.zeros((num_nodes, max_degree, max_k), dtype=np.int8)
            eids = np.zeros((num_nodes, max_degree), dtype=np.int32)

            for row, u in enumerate(gidxs.tolist()):
                write_row = 0
                # Iterate through edge size buckets (2-node edges, 3-node edges, etc.)
                for a in range(max_k):
                    for eid, others in zip(edge_ids[u][a], neighbors[u][a]):
                        L = len(others)
                        if L:
                            padded[row, write_row, :L] = np.asarray(
                                others, dtype=np.int32
                            )
                            mask[row, write_row, :L] = 1
                            eids[row, write_row] = eid
                            write_row += 1

            padded_adj_arrays.append(jnp.array(padded))
            mask_arrays.append(jnp.array(mask, dtype=jnp.bool))
            edge_id_arrays.append(jnp.array(eids))

        return padded_adj_arrays, mask_arrays, edge_id_arrays

    def get_edge_structure(
        self,
    ) -> tuple[Int[Array, "num_edges max_k"], Int[Array, "num_edges max_k"]]:
        """
        Convert edge list to padded array representation.
        """
        max_k = max(len(edge.nodes) for edge in self.edges)
        num_edges = len(self.edges)

        edge_indices = np.zeros((num_edges, max_k), dtype=np.int32)
        edge_mask = np.zeros((num_edges, max_k), dtype=np.bool_)

        for eid, edge in enumerate(self.edges):
            idxs = [self.node_to_global[v] for v in edge.nodes]
            k = len(idxs)
            edge_indices[eid, :k] = idxs
            edge_mask[eid, :k] = True

        return jnp.array(edge_indices), jnp.array(edge_mask)

    def get_sampling_params(
        self,
    ) -> tuple[
        tuple[
            list[Int[Array, "num_nodes max_edges max_k-1"]],
            list[Int[Array, "num_nodes max_edges max_k-1"]],
            list[Int[Array, "num_nodes max_edges"]],
        ],
        EqxGraph,
    ]:
        """
        Generate all parameters needed for graph sampling.
        """
        return self.get_padded_adj_arr(), EqxGraph(
            self.node_to_global, self.node_to_local, self.block_to_global
        )
