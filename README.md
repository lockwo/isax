# isax

isax is a [JAX](https://github.com/google/jax)-based library for sampling from Ising models using blocked Gibbs sampling. It supports hypergraphs, flexible sampling/modeling, and all the usual JAX transformations. isax is heavily inspired by [thrml](https://docs.thrml.ai/en/latest/) and [isingtorch](https://github.com/lockwo/isingtorch).

## Documentation

Available at https://lockwo.github.io/isax

## Installation

```bash
git clone https://github.com/lockwo/isax
cd isax
pip install -e .
```

Requires Python 3.10+.

## Quick Example

```python
import jax
import jax.numpy as jnp
from isax import BlockGraph, Edge, Node, IsingModel, IsingSampler, SamplingArgs, sample_chain

L = 4
nodes = [Node() for _ in range(L * L)]

edges = []
for x in range(L):
    for y in range(L):
        i = x * L + y
        edges.append(Edge(nodes[i], nodes[(x * L + (y + 1) % L)]))
        edges.append(Edge(nodes[i], nodes[((x + 1) % L) * L + y]))

even = [nodes[x * L + y] for x in range(L) for y in range(L) if (x + y) % 2 == 0]
odd = [nodes[x * L + y] for x in range(L) for y in range(L) if (x + y) % 2 == 1]

graph = BlockGraph([even, odd], edges)
params = graph.get_sampling_params()

model = IsingModel(weights=jnp.ones(len(edges)), biases=jnp.zeros(L * L))
sampler = IsingSampler()
sampling_args = SamplingArgs(gibbs_steps=100, blocks_to_sample=[0, 1], data=params)

key = jax.random.key(0)
init_state = [jax.random.choice(key, jnp.array([-1, 1]), (len(even),)),
              jax.random.choice(key, jnp.array([-1, 1]), (len(odd),))]

samples = sample_chain(init_state, [sampler, sampler], model, sampling_args, key)
```


## Future Additions (TODO)

- [x] annealing
- [x] cleaner interface
- [ ] improve example documentation/math background
- [ ] add tests
- [ ] runtime sampling params
- [ ] generic block typing
- [ ] generalize pytree typing for states
- [ ] support non-gibbs samplers (wolff, mh, etc.)
- [ ] add some ML examples