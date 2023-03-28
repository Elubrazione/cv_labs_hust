from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import optim
from arch_layers import ResNet
from draw_fig import draw
import torch.nn.functional as F
import numpy as np
import torch
import os

MODEL = False
EPOCHS = 100
BATCH_SIZE = 128
DEVICE = 'cpu'


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
  if not os.path.exists('./lab2/results'):
    os.makedirs('./lab2/results')

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  train_dataset = CIFAR10(root='./lab2/dataset', train=True, download=True, transform=transform_train)
  test_dataset = CIFAR10(root='./lab2/dataset', train=False, download=True, transform=transform_test)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

  model = ResNet().to(DEVICE)
  if MODEL:
    model.load_state_dict(torch.load('./lab2/model/model.pth'))
  optimizer = optim.Adam(model.parameters())

  accs = []
  for epoch in range(1, EPOCHS+1):
    print(f'epoch: {epoch}')
    train(model, train_loader, optimizer)
    correct, acc = test(model, test_loader)
    accs.append(acc)
    with open(f'./lab2/results_2.txt', "a") as f:
      f.write(f'epoch{epoch}  ' + f'acc {acc}' + '\n' + str(correct) + '\n')

    if epoch % 5 == 0:
      torch.save(model.state_dict(), f'./lab2/model/model_{epoch}.pth')
  draw(accs)