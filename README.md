# Critical Momenta Optimizers

Critical Momenta (CM) optimizers from the "Promoting Exploration in Memory-Augmented Adam
using Critical Momenta" project and [paper](https://arxiv.org/abs/), reformatted as package and stripped down to just the necessary components to integrate the optimizers into your code.
We also provide a faster and efficient GPU-based version of Critical Gradients optimizer from the paper from the "Memory Augmented Optimizers for Deep Learning" project and [paper](https://arxiv.org/abs/2106.10708).

This code is compatible with the following versions:

```
python >= 3.6
pytorch >= 1.7.1
```

## Installation

Clone [this repository](https://github.com/chandar-lab/CMOptimizer) anywhere on your system. Once cloned, `cd` to the directory and install with: `pip install .`

## Importing and Running

You can import the optimizers as you would any PyTorch optimizer. There are no requirements to run them other than PyTorch and its dependencies.

When installed, import the optimizers to your training script as needed:

```
from cmoptimizer import Adam_CM
```

You can then replace any PyTorch optimizer in your script with their `_CM` counterpart. Note that currently only Critical-Momenta variants of Adam and Critical-Gradient variants of Adam, RMSprop and SGD are implemented.

Here is a sample replacement:

```
optimizer = Adam(model.parameters(), lr=0.001)
```

becomes

```
optimizer = Adam_CM(model.parameters(), lr=0.001, **kwargs)
```

Similarly, for efficient GPU-based implementation of Critical gradients:  

```
from cmoptimizer.optim import SGD_C, RMSprop_C, Adam_C
optimizer = Adam_C(model.parameters(), lr=0.001, **kwargs)
```

## Optimizer Usage and Tuning

The Critical Momenta variants use all the same hyperparameters as their vanilla counterparts, so you may not need to perform any additional tuning.

The `_CM` optimizers have two additional hyperparameters compared to the vanilla version: `topC` which indicates how many critical momenta to keep and`decay` which indicates how much the norms of corresponding gradients are decayed each step. These are keyword arguments with default values which we observed to work well. For additional performance, these can be tuned.

The `_CM` variants perform best using either the same best learning rate as its vanilla counterpart, or 1/10 that learning rate. It is recommended you run both learning rates to compare.

Hyperparameter  `topC` determines how many critical gradients are stored and thus how much memory is used. Higher `topC` usually result in longer training times. Good `topC` values usually fall between 5 and 20. We recommended using values 5, 10, and 20.

Hyperparameter `decay` indicates the level of decay in the buffer. This modifies how frequently the buffer is refreshed. The `decay` parameter must fall between 0 and 1. We recommended using values 0.7 and 0.99.

[//]: # (## Citation)

[//]: # ()
[//]: # (```)

[//]: # (@misc{malviya021memory,)

[//]: # (  author    = {McRae, Paul-Aymeric and Parthasarathi, Prasanna and Assran, Mahmoud and Chandar, Sarath},)

[//]: # (  title     = {Memory Augmented Optimizers for Deep Learning},)

[//]: # (  year      = {2022},)

[//]: # (  booktitle = {Proceedings of ICLR})

[//]: # (})
