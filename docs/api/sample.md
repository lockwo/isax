# Sampling

## Models

::: isax.sample.IsingModel
    options:
        members:
            - energy
            - to_sample_params

## Samplers

::: isax.sample.AbstractSampler
    options:
        members: false

::: isax.sample.IsingSampler
    options:
        members:
            - sample
            - initialize_state

::: isax.sample.AnnealedIsingSampler
    options:
        members:
            - __init__
            - sample
            - initialize_state

## Sampling Utilities

::: isax.sample.SamplingArgs
    options:
        members:
            - __init__

::: isax.sample.sample_chain

