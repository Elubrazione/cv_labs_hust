from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import optim, nn
from arch_layers import ConvNet
import os

BATCH_SIZE = 64


if __name__ == '__main__':
  if not os.path.exists('./lab2/dataset'):
    os.makedirs('./lab2/dataset')

  cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  train_dataset = CIFAR10(root='./lab2/dataset', train=True, download=True, transform=cifar_transform)
  test_dataset = CIFAR10(root='./lab2/dataset', train=False, download=True, transform=cifar_transform)

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

  model = ConvNet()
  optimizer = optim.Adam(model.parameters())
  loss_fc = nn.CrossEntropyLoss()
