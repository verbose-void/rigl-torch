"""


Note: This is the exact same script as found here: https://github.com/pytorch/examples/blob/0f0c9131ca5c79d1332dce1f4c06fe942fbdc665/mnist/main.py#L1
The only difference is there are a few parser arguments added and the mandatory rigl-torch code (creating the prune scheduler).


"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from rigl_torch.RigL import RigLScheduler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, pruner):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
    
        if pruner():
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def ed(param_name, default=None):
    return os.environ.get(param_name, default)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dense-allocation', default=ed('DENSE_ALLOCATION'), type=float,
                        help='percentage of dense parameters allowed. if None, pruning will not be used. must be on the interval (0, 1]')
    parser.add_argument('--delta', default=ed('DELTA', 100), type=int,
                        help='delta param for pruning')
    parser.add_argument('--grad-accumulation-n', default=ed('GRAD_ACCUMULATION_N', 1), type=int,
                        help='number of gradients to accumulate before scoring for rigl')
    parser.add_argument('--alpha', default=ed('ALPHA', 0.3), type=float,
                        help='alpha param for pruning')
    parser.add_argument('--static-topo', default=ed('STATIC_TOPO', 0), type=int, help='if 1, use random sparsity topo and remain static')
    parser.add_argument('--batch-size', type=int, default=ed('BATCH_SIZE', 64), metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=ed('TEST_BATCH_SIZE', 1000), metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=ed('EPOCHS', 14), metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=ed('LR', 1), metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=ed('GAMMA', 0.7), metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', default=1, type=bool,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if args.dense_allocation is None:
        print('-------------------------------------------------------------------')
        print('heads up, RigL will not be used unless `--dense-allocation` is set!')
        print('-------------------------------------------------------------------')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    pruner = lambda: True
    if args.dense_allocation is not None:
        T_end = int(0.75 * args.epochs * len(train_loader))
        pruner = RigLScheduler(model, optimizer, dense_allocation=args.dense_allocation, alpha=args.alpha, delta=args.delta, static_topo=args.static_topo, T_end=T_end, ignore_linear_layers=False, grad_accumulation_n=args.grad_accumulation_n)

    writer = SummaryWriter(log_dir='./graphs')

    print(model)
    for epoch in range(1, args.epochs + 1):
        print(pruner)
        train(args, model, device, train_loader, optimizer, epoch, pruner=pruner)
        loss, acc = test(model, device, test_loader)
        scheduler.step()

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('accuracy', acc, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "/artifacts/mnist_cnn.pt")


if __name__ == '__main__':
    main()
