import torch
import torchvision


def get_conv_layers_with_activations(model, i=0, layers=None, linear_layers_mask=None):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    for layer_name, p in model._modules.items():
        if isinstance(p, torch.nn.Conv2d):
            layers.append([p])
            linear_layers_mask.append(0)
            i += 1
        # elif isinstance(p, torch.nn.AdaptiveAvgPool2d):
            # layers.append([p])
            # i += 1
        elif isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
        elif isinstance(p, torch.nn.BatchNorm2d):
            layers[-1].append(p)
        elif isinstance(p, torch.nn.ReLU):
            layers[-1].append(p)
        elif isinstance(p, torch.nn.MaxPool2d) or isinstance(p, torch.nn.AdaptiveAvgPool2d):
            layers[-1].append(p)
        elif layer_name == 'downsample':
            layers.append(p)
            linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(p, torchvision.models.resnet.BasicBlock):
            # if hasattr(p, 'downsample') and p.downsample is not None:
                # identity_indices.append(i)
            _, linear_layers_mask, i = get_conv_layers_with_activations(p, i=i, layers=layers, linear_layers_mask=linear_layers_mask)
        else:
            _, linear_layers_mask, i = get_conv_layers_with_activations(p, i=i, layers=layers, linear_layers_mask=linear_layers_mask)

    return layers, linear_layers_mask, i 


def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_conv_layers_with_activations(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W
