# Sampling

## Models

::: isax.IsingModel
    options:
        members:
            - energy
            - to_sample_params

## Samplers

::: isax.AbstractSampler
    options:
        members: false

::: isax.IsingSampler
    options:
        members:
            - sample
            - initialize_state

::: isax.AnnealedIsingSampler
    options:
        members:
            - __init__
            - sample
            - initialize_state

## Sampling Utilities

::: isax.SamplingArgs
    options:
        members:
            - __init__

::: isax.sample_chain
