from abc import abstractmethod
from typing import Callable, Generic, TypeVar

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree

from .block import EqxGraph


_State = TypeVar("_State")


class IsingModel(eqx.Module):
    r"""Ising model with arbitrary edge interactions and local fields."""

    weights: Float[Array, " num_edges"]
    biases: Float[Array, " num_nodes"]

    def energy(
        self,
        state: Int[Array, " num_nodes"],
        edge_indices: Int[Array, "num_edges max_k"],
        edge_mask: Int[Array, "num_edges max_k"],
    ) -> Float[Array, ""]:
        r"""Compute Ising energy.

        $$H = -\sum_{e \in E} J_e \prod_{i \in e} s_i - \sum_i h_i s_i$$

        **Arguments:**

        - `state`: Node spins, shape `(num_nodes,)`.
        - `edge_indices`: Node indices for each edge, shape `(num_edges, max_k)`.
        - `edge_mask`: Valid positions in `edge_indices`.

        **Returns:**

        Scalar energy.
        """
        edge_spins = state[edge_indices]
        # Replace padded positions with 1 (multiplicative identity)
        masked_spins = jnp.where(edge_mask, edge_spins, 1)
        spin_products = jnp.prod(masked_spins, axis=-1)
        edge_energy = jnp.sum(self.weights * spin_products)
        field_energy = jnp.sum(self.biases * state)
        return -(edge_energy + field_energy)

    def to_sample_params(
        self, graph: EqxGraph, edge_info: list[Int[Array, "nodes max_edges"]]
    ) -> list[tuple[Float[Array, " num_edges"], Float[Array, " num_nodes"]]]:
        """Extract weights and biases for each block.

        **Arguments:**

        - `graph`: Graph structure.
        - `edge_info`: Edge indices per block.

        **Returns:**

        List of `(edge_weights, node_biases)` tuples.
        """
        params = []
        # could be called in sampling if memory too large
        for i, block in enumerate(graph.block_to_global):
            # Get weights for edges incident to this block
            # and biases for nodes in this block
            params.append((self.weights[edge_info[i]], self.biases[block]))
        return params


class AbstractSampler(eqx.Module, Generic[_State]):
    """Base class for block-wise sampling algorithms."""

    @abstractmethod
    def sample(
        self,
        current_state: Int[Array, " num_nodes"],
        neighbor_states: Int[Array, "num_nodes max_edges max_k-1"],
        neighbor_mask: Int[Array, "num_nodes max_edges max_k-1"],
        model_params: tuple[Float[Array, " num_edges"], Float[Array, " num_nodes"]],
        runtime_params: PyTree,
        sampler_state: _State,
        key: Key[Array, ""],
    ) -> tuple[Int[Array, " num_nodes"], _State]:
        """Sample new configuration for a block given its neighbors.

        **Arguments:**

        - `current_state`: Current spin values for nodes in this block.
        - `neighbor_states`: Spin values of neighbors from all edges.
        - `neighbor_mask`: Valid positions in neighbor_states.
        - `model_params`: Tuple of `(edge_weights, node_biases)` for this block.
        - `runtime_params`: Additional runtime parameters.
        - `sampler_state`: Samplere state.
        - `key`: Random key for stochastic sampling.

        **Returns:**

        Tuple of `(new_block_state, updated_sampler_state)`.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_state(self) -> _State:
        """Initialize the sampler state."""
        raise NotImplementedError


class IsingSampler(AbstractSampler[None]):
    """Gibbs sampler for Ising model at fixed temperature ($\\beta=1$)."""

    def sample(
        self,
        current_state: Int[Array, " num_nodes"],
        neighbor_states: Int[Array, "num_nodes max_edges max_k-1"],
        neighbor_mask: Int[Array, "num_nodes max_edges max_k-1"],
        model_params: tuple[Float[Array, " num_edges"], Float[Array, " num_nodes"]],
        runtime_params: None,
        sampler_state: None,
        key: Key[Array, ""],
    ) -> tuple[Int[Array, " num_nodes"], None]:
        r"""Parallel Gibbs sampling for all nodes in block.

        For each node $i$, the effective field is computed as:

        $$h_i^{\text{eff}} = h_i + \sum_{e \ni i} J_e \prod_{j \in e \setminus i} s_j$$

        The probability of spin $s_i = +1$ is then:

        $$P(s_i = +1) = \sigma(2h_i^{\text{eff}}) =
        \frac{1}{1 + e^{-2h_i^{\text{eff}}}}$$

        where $\sigma$ is the sigmoid function.

        **Arguments:**

        - `current_state`: Current spin values for nodes in this block.
        - `neighbor_states`: Spin values of neighbors from all edges.
        - `neighbor_mask`: Boolean mask for valid positions in `neighbor_states`.
        - `model_params`: Tuple of `(edge_weights, node_biases)` for this block.
        - `runtime_params`: None.
        - `sampler_state`: None.
        - `key`: JAX random key for stochastic sampling.

        **Returns:**

        Tuple of `(new_block_state, None)` where new_block_state contains the sampled
        spin configuration.
        """
        edge_weights, biases = model_params
        prod_neighbors = jnp.prod(jnp.where(neighbor_mask, neighbor_states, 1), axis=-1)
        edge_mask = jnp.any(neighbor_mask, axis=-1)
        field = biases + jnp.sum(edge_weights * prod_neighbors * edge_mask, axis=-1)
        probs = jax.nn.sigmoid(2.0 * field)
        new_state = (
            jax.random.bernoulli(key, p=probs, shape=current_state.shape).astype(
                current_state.dtype
            )
            * 2
        ) - 1
        return new_state, None

    def initialize_state(self) -> None:
        """Return None for stateless sampler."""
        return None


class AnnealedIsingSampler(AbstractSampler[Int[Array, ""]]):
    """Simulated annealing sampler with time-varying temperature."""

    beta_fn: Callable[[Int[Array, ""]], Float[Array, ""]]

    def __init__(self, beta_fn: Callable[[Int[Array, ""]], Float[Array, ""]]):
        r"""**Arguments:**

        - `beta_fn`: Function mapping timestep $t$ to inverse temperature $\beta(t)$.
            Can also curry beta arrays for more complex schedules.
        """
        self.beta_fn = beta_fn

    def sample(
        self,
        current_state: Int[Array, " num_nodes"],
        neighbor_states: Int[Array, "num_nodes max_edges max_k-1"],
        neighbor_mask: Int[Array, "num_nodes max_edges max_k-1"],
        model_params: tuple[Float[Array, " num_edges"], Float[Array, " num_nodes"]],
        runtime_params: None,
        sampler_state: Int[Array, ""],
        key: Key[Array, ""],
    ) -> tuple[Int[Array, " num_nodes"], Int[Array, ""]]:
        """Sample with temperature based on current timestep.

        **Arguments:**

        - `current_state`: Current spin values for nodes in this block.
        - `neighbor_states`: Spin values of neighbors from all edges.
        - `neighbor_mask`: Boolean mask for valid positions in `neighbor_states`.
        - `model_params`: Tuple of `(edge_weights, node_biases)` for this block.
        - `runtime_params`: Unused, provided for API compatibility.
        - `sampler_state`: Current timestep (integer scalar).
        - `key`: JAX random key for stochastic sampling.

        **Returns:**

        Tuple of `(new_block_state, new_sampler_state)` where new_sampler_state is
        incremented by 1.
        """
        beta = self.beta_fn(sampler_state)

        edge_weights, biases = model_params
        prod_neighbors = jnp.prod(jnp.where(neighbor_mask, neighbor_states, 1), axis=-1)
        edge_mask = jnp.any(neighbor_mask, axis=-1)
        field = biases + jnp.sum(edge_weights * prod_neighbors * edge_mask, axis=-1)

        probs = jax.nn.sigmoid(2.0 * beta * field)
        new_state = (
            jax.random.bernoulli(key, p=probs, shape=current_state.shape).astype(
                current_state.dtype
            )
            * 2
        ) - 1

        new_sampler_state = sampler_state + 1

        return new_state, new_sampler_state

    def initialize_state(self) -> Int[Array, ""]:
        """Initialize time step to 0."""
        return jnp.array(0, dtype=jnp.int32)


class SamplingArgs(eqx.Module):
    """Configuration and data for block-wise Gibbs sampling."""

    gibbs_steps: int
    blocks_to_sample: list[int]
    adjs: list[Int[Array, "nodes max_edges max_k-1"]]
    masks: list[Int[Array, "nodes max_edges max_k-1"]]
    edge_ids: list[Int[Array, "nodes max_edges"]]
    eqx_graph: EqxGraph

    def __init__(
        self,
        gibbs_steps: int,
        blocks_to_sample: list[int],
        data: tuple[tuple, EqxGraph],
    ) -> None:
        """**Arguments:**

        - `gibbs_steps`: Number of Gibbs sampling iterations to perform.
        - `blocks_to_sample`: List of block indices to update (e.g., `[0, 2]` updates
            blocks 0 and 2).
        - `data`: Output from `BlockGraph.get_sampling_params()` containing adjacency
            information and graph structure.
        """
        self.gibbs_steps = gibbs_steps
        self.blocks_to_sample = blocks_to_sample
        (self.adjs, self.masks, self.edge_ids), self.eqx_graph = data


def concat_state(states: list[Int[Array, " block_size"]]) -> Int[Array, " total_size"]:
    """Concatenate block states into global state array."""
    if states[0].shape == ():
        return jnp.stack(states)
    return jnp.concatenate(states, axis=0)


def sample_blocks(
    block_states: list[Int[Array, " block_size"]],
    sampler_states: list[PyTree],
    samplers: list[AbstractSampler],
    params: list[tuple[Float[Array, " num_edges"], Float[Array, " num_nodes"]]],
    sampling_args: SamplingArgs,
    key: Key[Array, ""],
) -> tuple[list[Int[Array, " block_size"]], list[PyTree]]:
    """Sequentially update specified blocks using their samplers."""
    for i in sampling_args.blocks_to_sample:
        neighbors = jax.tree.map(
            lambda x: x[sampling_args.adjs[i]], concat_state(block_states)
        )
        new_state, new_sampler_state = samplers[i].sample(
            block_states[i],
            neighbors,
            sampling_args.masks[i],
            params[i],
            None,
            sampler_states[i],
            key,
        )
        block_states[i] = new_state
        sampler_states[i] = new_sampler_state
        key = jax.random.fold_in(key, i)

    return block_states, sampler_states


def sample_chain(
    block_states: list[Int[Array, " block_size"]],
    samplers: list[AbstractSampler],
    model: IsingModel,
    sampling_args: SamplingArgs,
    key: Key[Array, ""],
) -> list[Int[Array, "gibbs_steps block_size"]]:
    """Run Gibbs sampling chain for specified number of steps.

    **Arguments:**

    - `block_states`: Initial spin states for each block.
    - `samplers`: List of sampler instances, one per block.
    - `model`: Energy model providing weights and biases.
    - `sampling_args`: Configuration with number of steps and blocks to sample.
    - `key`: JAX random key.

    **Returns:**

    List of arrays containing the history of block states over all Gibbs steps.
    """
    keys = jax.random.split(key, sampling_args.gibbs_steps)
    params = model.to_sample_params(sampling_args.eqx_graph, sampling_args.edge_ids)

    sampler_states = [sampler.initialize_state() for sampler in samplers]

    def _sample(carry, key):
        states, sampler_states = carry
        new_states, new_sampler_states = sample_blocks(
            states, sampler_states, samplers, params, sampling_args, key
        )
        return (new_states, new_sampler_states), new_states

    _, block_states_history = jax.lax.scan(
        _sample, (block_states, sampler_states), keys
    )
    return block_states_history
