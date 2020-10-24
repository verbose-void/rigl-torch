import torch

from rigl_torch.RigL import RigLScheduler


# set up environment
torch.manual_seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
max_iters = 50000
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(device)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)

# hyperparameters
T_end = int(max_iters * 0.75)
delta = 2
dense_allocation = 0.1



class TestGrowDrop:
    def test_initial_sparsity(self):
        scheduler = RigLScheduler(model, optimizer, dense_allocation=dense_allocation, T_end=T_end, delta=delta)

        for l, (W, target_S, N) in enumerate(zip(scheduler.W, scheduler.S, scheduler.N)):
            target_zeros = int(target_S * N)
            actual_zeros = torch.sum(W == 0).item()
            assert actual_zeros == target_zeros, '%i != %i' % (actual_zeros, target_zeros)

    def test_mask(self):
        pass
