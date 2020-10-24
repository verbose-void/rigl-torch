import torch

from rigl_torch.RigL import RigLScheduler


# set up environment
torch.manual_seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyperparameters
arch = 'resnet50'
image_dimensionality = (3, 224, 224)
num_classes = 1000
max_iters = 64
T_end = int(max_iters * 0.75)
delta = 2
dense_allocation = 0.1
criterion = torch.nn.functional.cross_entropy


def get_dummy_dataloader():
    X = torch.rand((max_iters, *image_dimensionality))
    T = (torch.rand(max_iters) * num_classes).long()
    dataset = torch.utils.data.TensorDataset(X, T)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    return dataloader


def get_new_scheduler():
    model = torch.hub.load('pytorch/vision:v0.6.0', arch, pretrained=False).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
    scheduler = RigLScheduler(model, optimizer, dense_allocation=dense_allocation, T_end=T_end, delta=delta)
    print(model)
    print(scheduler)
    return scheduler


def assert_actual_sparsity_is_valid(scheduler):
    for l, (W, target_S, N) in enumerate(zip(scheduler.W, scheduler.S, scheduler.N)):
        target_zeros = int(target_S * N)
        actual_zeros = torch.sum(W == 0).item()
        print('----- layer %i ------' % l)
        print('target_zeros', target_zeros)
        print('actual_zeros', actual_zeros)
        assert actual_zeros == target_zeros


class TestRigLScheduler:
    def test_initial_sparsity(self):
        scheduler = get_new_scheduler()
        assert_actual_sparsity_is_valid(scheduler)


    def test_sparse_elements_are_zeros(self):
        scheduler = get_new_scheduler()
        model = scheduler.model
        optimizer = scheduler.optimizer
        dataloader = get_dummy_dataloader()

        model.train()
        for i, (X, T) in enumerate(dataloader):
            Y = model(X.to(device))
            loss = criterion(Y, T.to(device))
            loss.backward()

            if scheduler():
                optimizer.step()

            print('iteration %i' % i)
            assert_actual_sparsity_is_valid(scheduler)
    
