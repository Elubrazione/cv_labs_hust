from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import optim
from arch_layers import ConvNet
import torch.nn.functional as F
import numpy as np
import torch
import time
import os

EPOCHS = 20
BATCH_SIZE = 64
DEVICE = 'cpu'
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


def train(model, train_loader, optimizer):
  model.train()
  for _, (data, target) in enumerate(train_loader):
    data, target = data.to(torch.float32).to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def test(model, test_loader):
  model.eval()
  losses = 0.0
  correct = [0 for i in range(10)]
  acc = 0.0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(torch.float32), target.to(DEVICE)
      output = model(data)
      losses += F.cross_entropy(output, target, reduction='sum').item()
      _, prediction = torch.max(output, 1, keepdim=True)

      target = np.array(target).tolist()
      prediction = np.array(prediction).T.tolist()[0]

      for i in range(len(target)):
        # print(prediction[i], target[i])
        if prediction[i] == target[i]:
          acc += 1
          correct[prediction[i]] += 1
  ave = acc / 10000
  print(ave*100, correct)
  return correct, ave*100


if __name__ == '__main__':
  if not os.path.exists('./lab2/dataset'):
    os.makedirs('./lab2/dataset')
  if not os.path.exists('./lab2/model'):
    os.makedirs('./lab2/model')

  cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  train_dataset = CIFAR10(root='./lab2/dataset', train=True, download=True, transform=cifar_transform)
  test_dataset = CIFAR10(root='./lab2/dataset', train=False, download=True, transform=cifar_transform)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

  model = ConvNet().to(DEVICE)
  optimizer = optim.Adam(model.parameters())

  accs = []
  for epoch in range(1, EPOCHS+1):
    print(f'epoch: {epoch}')
    train(model, train_loader, optimizer)
    correct, acc = test(model, test_loader)
    accs.append(acc)
    with open(f'./lab2/results.txt', "a") as f:
      f.write(f'epoch{epoch}  ' + f'acc {acc}' + '\n' + str(correct) + '\n')

  torch.save(model, f'./lab2/model/{time.time()}.pkl')