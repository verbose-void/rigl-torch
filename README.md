# rigl-torch
![PyTest](https://github.com/McCrearyD/rigl-torch/workflows/PyTest/badge.svg)

## Warning: This repository is still in active development, results are not yet up to the rigl paper spec. Coming soon!

An open source implementation of Google Research's paper (Authored by [Utku Evci](https://www.linkedin.com/in/utkuevci/), an AI Resident @ Google Brain):  [Rigging the Lottery: Making All Tickets Winners (RigL)](https://arxiv.org/abs/1911.11134) in PyTorch as versatile, simple, and fast as possible.

You only need to add ***2 lines of code*** to your PyTorch project to use RigL to train your model with sparsity!

## Other Implementations:
- View the TensorFlow implementation (also the original) [here](https://github.com/google-research/rigl)!
- Additionally, it is also implemented in [vanilla python](https://evcu.github.io/ml/sparse-micrograd/) and [graphcore](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl).

## Setup:
- Clone this repository: `git clone https://github.com/McCrearyD/rigl-torch`
- Cd into repo: `cd rigl-torch`
- Install dependencies: `pip install -r requirements.txt`
- Install package (`-e` allows for modifications): `pip install -e .`

## Usage:
- Run the tests by doing `cd rigl-torch`, then `pytest`.

- I have provided some examples of training scripts that were **slightly** modified to add RigL's functionality. It adds a few parser statements, and only 2 required lines of RigL code usage to work! See them with links to the originals here:
    - `ImageNet` | [RigL](https://github.com/McCrearyD/rigl-pytorch/blob/master/train_imagenet_rigl.py) | [Original](https://github.com/pytorch/examples/blob/0f0c9131ca5c79d1332dce1f4c06fe942fbdc665/imagenet/main.py#L1) | [RigL + SageMaker](https://github.com/McCrearyD/rigl-pytorch/blob/master/sagemaker/rigl.ipynb)
    - `MNIST` | [RigL](https://github.com/McCrearyD/rigl-pytorch/blob/master/train_mnist_rigl.py) | [Original](https://github.com/pytorch/examples/blob/0f0c9131ca5c79d1332dce1f4c06fe942fbdc665/mnist/main.py#L1)
  
- OR more impressively, **you can use the pruning power of RigL by adding 2 lines of code to your already existing training script**! Here is how:

```python
from rigl_torch.RigL import RigLScheduler

# first, create your model
model = ... # note: only tested on torch.hub's resnet networks (ie. resnet18 / resnet50)

# create your dataset/dataloader
dataset = ...
dataloader = ...

# define your optimizer (recommended SGD w/ momentum)
optimizer = ...


# RigL runs best when you allow RigL's topology modifications to run for 75% of the total training iterations (batches)
# so, let's calculate T_end according to this
epochs = 100
total_iterations = len(dataloader) * epochs
T_end = int(0.75 * total_iterations)

# ------------------------------------ REQUIRED LINE # 1 ------------------------------------
# now, create the RigLScheduler object
pruner = RigLScheduler(model,                  # model you created
                       optimizer,              # optimizer (recommended = SGD w/ momentum)
                       dense_allocation=0.1,   # a float between 0 and 1 that designates how sparse you want the network to be (0.1 dense_allocation = 90% sparse)
                       T_end=T_end,            # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                       delta=100,              # delta hyperparam within the paper (recommended = 100)
                       alpha=0.3,              # alpha hyperparam within the paper (recommended = 0.3)
                       static_topo=False)      # if True, the topology will be frozen, in other words RigL will not do it's job (for debugging)
# -------------------------------------------------------------------------------------------
                       
... more code ...

for data in dataloader:
    # do forward pass, calculate loss, etc.
    ...
    
    # instead of calling optimizer.step(), wrap it as such:
    
# ------------------------------------ REQUIRED LINE # 2 ------------------------------------
    if pruner():
# -------------------------------------------------------------------------------------------
        # this block of code will execute according to the given hyperparameter schedule
        # in other words, optimizer.step() is not called after a RigL step
        optimizer.step()
        
# at any time you can print the RigLScheduler object and it will show you the sparsity distributions, number of training steps/rigl steps, etc!
print(pruner)

# save model
torch.save(model)
```

