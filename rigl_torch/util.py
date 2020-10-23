import torch
import torchvision


def get_conv_layers_with_activations(model, i=0, layers=None, identity_indices=None, ignore_linear_layers=False):
    if layers is None:
        layers = []
    if identity_indices is None:
        identity_indices = []

    for layer_name, p in model._modules.items():
        if isinstance(p, torch.nn.Conv2d):
            layers.append([p])
            i += 1
        # elif isinstance(p, torch.nn.AdaptiveAvgPool2d):
            # layers.append([p])
            # i += 1
        elif isinstance(p, torch.nn.Linear) and not ignore_linear_layers:
            layers.append([p])
        elif isinstance(p, torch.nn.BatchNorm2d):
            layers[-1].append(p)
        elif isinstance(p, torch.nn.ReLU):
            layers[-1].append(p)
        elif isinstance(p, torch.nn.MaxPool2d) or isinstance(p, torch.nn.AdaptiveAvgPool2d):
            layers[-1].append(p)
        elif layer_name == 'downsample':
            layers.append(p)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(p, torchvision.models.resnet.BasicBlock):
            if hasattr(p, 'downsample') and p.downsample is not None:
                identity_indices.append(i)
            _, identity_indices, i = get_conv_layers_with_activations(p, i=i, layers=layers, identity_indices=identity_indices)
        else:
            _, identity_indices, i = get_conv_layers_with_activations(p, i=i, layers=layers, identity_indices=identity_indices)

    return layers, identity_indices, i 


def get_W(model, ignore_linear_layers=False):
    layers, _, _ = get_conv_layers_with_activations(model, ignore_linear_layers=ignore_linear_layers)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)
    return W
