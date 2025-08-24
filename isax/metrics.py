import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


def magnetization_per_site(
    state: list[Int[Array, "samples block_nodes"]],
) -> list[Float[Array, " block_nodes"]]:
    r"""Compute magnetization per site.

    $$m_j = \frac{1}{N} \sum_{i=1}^{N} s_{i,j}$$

    **Arguments:**

    - `state`: List of arrays, shape `(samples, block_nodes)` per block.

    **Returns:**

    List of arrays, shape `(block_nodes,)` per block.
    """
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), state)
