
from rigl_torch.RigL import RigLScheduler
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

parser = argparse.ArgumentParser(description='Test for analyzing the consistency of sparsity values throughout training.')
parser.add_argument('--data-path', help='path to dataset', required=True)
args = parser.parse_args()

torch.manual_seed(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

max_iters = 50000

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(device)
# model = torch.nn.DataParallel(model).to(device)
scheduler = RigLScheduler(model, dense_allocation=0.1, T_end=int(max_iters * 0.75), delta=2)

print(scheduler)
print('---------------------------------------')

dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)#, num_workers=24)

# calculate gradients once (so we can do the tests without failure)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1)
model.train()

for epoch in range(5):
    print('EPOCH [%i]' % epoch)
    for i, (X, T) in enumerate(dataloader):
        optimizer.zero_grad()
        Y = model(X.to(device))
        loss = criterion(Y, T.to(device))
        loss.backward()
    
        if scheduler():
            optimizer.step()
            print('[iter %i]' % i)
        else:
            print('[iter %i] RIGL STEP' % i)
    
        if i % 50 == 0:
            print(scheduler)

    if i > max_iters:
        break
