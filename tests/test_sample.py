import unittest

import jax
import jax.numpy as jnp
from isax import (
    BlockGraph,
    Edge,
    IsingModel,
    IsingSampler,
    Node,
    sample_chain,
    SamplingArgs,
)


class TestSampleChain(unittest.TestCase):
    def test_readme_example(self):
        """Run the README example to ensure it works."""
        L = 4
        nodes = [Node() for _ in range(L * L)]

        edges = []
        for x in range(L):
            for y in range(L):
                i = x * L + y
                edges.append(Edge(nodes[i], nodes[(x * L + (y + 1) % L)]))
                edges.append(Edge(nodes[i], nodes[((x + 1) % L) * L + y]))

        even = [
            nodes[x * L + y] for x in range(L) for y in range(L) if (x + y) % 2 == 0
        ]
        odd = [nodes[x * L + y] for x in range(L) for y in range(L) if (x + y) % 2 == 1]

        graph = BlockGraph([even, odd], edges)
        params = graph.get_sampling_params()

        model = IsingModel(weights=jnp.ones(len(edges)), biases=jnp.zeros(L * L))
        sampler = IsingSampler()
        sampling_args = SamplingArgs(
            gibbs_steps=100, blocks_to_sample=[0, 1], data=params
        )

        key = jax.random.key(0)
        init_state = [
            jax.random.choice(key, jnp.array([-1, 1]), (len(even),)),
            jax.random.choice(key, jnp.array([-1, 1]), (len(odd),)),
        ]

        samples = sample_chain(
            init_state, [sampler, sampler], model, sampling_args, key
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].shape, (100, len(even)))
        self.assertEqual(samples[1].shape, (100, len(odd)))
        self.assertTrue(jnp.all((samples[0] == 1) | (samples[0] == -1)))
        self.assertTrue(jnp.all((samples[1] == 1) | (samples[1] == -1)))


if __name__ == "__main__":
    unittest.main()
