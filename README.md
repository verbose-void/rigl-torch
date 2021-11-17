# rigl-torch
![PyTest](https://github.com/McCrearyD/rigl-torch/workflows/PyTest/badge.svg)
![Upload Python Package](https://github.com/McCrearyD/rigl-torch/workflows/Upload%20Python%20Package/badge.svg)

An open source implementation of Google Research's paper (Authored by [Utku Evci](https://www.linkedin.com/in/utkuevci/), an AI Resident @ Google Brain):  [Rigging the Lottery: Making All Tickets Winners (RigL)](https://arxiv.org/abs/1911.11134) in PyTorch as versatile, simple, and fast as possible.

You only need to add ***2 lines of code*** to your PyTorch project to use RigL to train your model with sparsity!

## ImageNet Results
Results aren't quite as complete as the original paper, but if you end up running on ImageNet with different configurations and/or different datasets, I would love to include them in the repo here!

### RigL-1x:
| Architecture | Sparsity % | S. Distribution | Top-1 | [Original](https://github.com/google-research/rigl) Top-1 | Model/Ckpt |
| :------------- | :----------: | :-----------: | :-----------: | :-----------: | -----------: |
|  ResNet50 | 90%   | Uniform   | 72.4%    | 72%    | [Link](https://drive.google.com/file/d/1mzrS-V9RW6c2o3yj4N1BBp7Kp3z-Cjk8/view?usp=sharing) |

## Other Implementations:
- View the TensorFlow implementation (also the original) [here](https://github.com/google-research/rigl)!
- Additionally, it is also implemented in [vanilla python](https://evcu.github.io/ml/sparse-micrograd/) and [graphcore](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl).

## Contributions Beyond the Paper:
### Gradient Accumulation:
#### Motivation:
- [Evci et al.](https://arxiv.org/abs/1911.11134) cites their experiments for ImageNet being done using a batch size of 4096, which isn't practical for everyone since to do so you need 32 Tesla V100s to store that many 224x224x3 images in VRAM.
- Following this, if you are using a significantly small batch size for training (ie. bs=1024 for ImageNet), RigL may perform suboptimally due to instantaneous gradient information being quite noisy. To remedy this, I have introduced a solution for "emulating" larger batch sizes for topology modifications.
#### Method:
- In regular dense training gradients are calculated per batch essentially averaging the loss of each sample and taking the derivative w.r.t the parameters. This means that if your batch size is 1024, the gradients are the accumulated average over 1024 data samples.
- In normal RigL the grow/drop perturbations scoring is being done on 1 batch (every `delta` batches, typically `delta=100`) and replaces the backpropagation step for that iteration. So you can see that if the batch size is significantly small, the topology modifications are being done on a very small amount of data, thus missing some potential signal from the dataset. In dense training, this is a balancing act (too large batch sizes have diminishing returns and can harm exploration, making it more likely to fall in a local minimum).
- If `gradient_accumulation_n` is > 1, then when RigL wants to make a topology modification it essentially takes **not only** the current batch's gradients, but also the previous `gradient_accumulation_n - 1` batch's gradients. It then averages them element-wise, and uses this new matrix to score the grow/drop perturbations.
- **Note**: `gradient_accumulation_n` has to be within the interval \[1, `delta`). If `gradient_accumulation_n` == 1, then nothing has changed from the paper's spec. If `gradient_accumulation_n` == (`delta` - 1), RigL will score based on every single batch from the previous RigL step to the current one.
#### Results:
- Setting the `gradient_accumulation_n` to a value > 1 increases performance on ImageNet by about 0.3-1% when using a batch size of 1024. In order to get the best results from batch size 1024 (for ImageNet), **you should also multiply the `delta` value by 4**. This is because with a batch size of 4096, you are doing 4x less RigL steps (4096/1024 = 4) than if you used a batch size of 1024.

## User Setup:
- `pip install rigl-torch`
## Contributor Setup:
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
pruner = RigLScheduler(model,                           # model you created
                       optimizer,                       # optimizer (recommended = SGD w/ momentum)
                       dense_allocation=0.1,            # a float between 0 and 1 that designates how sparse you want the network to be 
                                                          # (0.1 dense_allocation = 90% sparse)
                       sparsity_distribution='uniform', # distribution hyperparam within the paper, currently only supports `uniform`
                       T_end=T_end,                     # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                       delta=100,                       # delta hyperparam within the paper (recommended = 100)
                       alpha=0.3,                       # alpha hyperparam within the paper (recommended = 0.3)
                       grad_accumulation_n=1,           # new hyperparam contribution (not in the paper) 
                                                          # for more information, see the `Contributions Beyond the Paper` section
                       static_topo=False,               # if True, the topology will be frozen, in other words RigL will not do it's job 
                                                          # (for debugging)
                       ignore_linear_layers=False,      # if True, linear layers in the network will be kept fully dense
                       state_dict=None)                 # if you have checkpointing enabled for your training script, you should save 
                                                          # `pruner.state_dict()` and when resuming pass the loaded `state_dict` into 
                                                          # the pruner constructor
# -------------------------------------------------------------------------------------------
                       
... more code ...

for epoch in range(epochs):
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
        
    # it is also recommended that after every epoch you checkpoint your training progress
    # to do so with RigL training you should also save the pruner object state_dict
    torch.save({
        'model': model.state_dict(),
        'pruner': pruner.state_dict(),
        'optimizer': optimizer.state_dict()
    }, 'checkpoint.pth')
        
# at any time you can print the RigLScheduler object and it will show you the sparsity distributions, number of training steps/rigl steps, etc!
print(pruner)

# save model
torch.save(model.state_dict(), 'model.pth')
```

## Citation
```
@misc{nollied, 
    author = {McCreary, Dyllan},
    title = {PyTorch Implementation of Rigging the Lottery: Making All Tickets Winners}, 
    url = {https://github.com/nollied/rigl-torch},
    year = {2020}, 
    month = {Nov},
    note = {Reimplementation/extension of the work done by Google Research: https://github.com/google-research/rigl}
}
```

## Used By
- [DCTpS Paper](https://arxiv.org/pdf/2102.07655.pdf)
