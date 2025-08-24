import unittest

import jax.numpy as jnp
from isax.metrics import magnetization_per_site


class TestMagnetizationPerSite(unittest.TestCase):
    def test_all_up_spins(self):
        """All +1 spins should give magnetization of 1."""
        state = [jnp.ones((10, 5), dtype=jnp.int32)]
        result = magnetization_per_site(state)
        self.assertEqual(len(result), 1)
        self.assertTrue(jnp.allclose(result[0], jnp.ones(5)))

    def test_all_down_spins(self):
        """All -1 spins should give magnetization of -1."""
        state = [jnp.full((10, 5), -1, dtype=jnp.int32)]
        result = magnetization_per_site(state)
        self.assertTrue(jnp.allclose(result[0], jnp.full(5, -1.0)))

    def test_balanced_spins(self):
        """Half +1 and half -1 should give magnetization of 0."""
        up = jnp.ones((5, 4), dtype=jnp.int32)
        down = jnp.full((5, 4), -1, dtype=jnp.int32)
        state = [jnp.concatenate([up, down], axis=0)]
        result = magnetization_per_site(state)
        self.assertTrue(jnp.allclose(result[0], jnp.zeros(4)))


if __name__ == "__main__":
    unittest.main()
