from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from arch_layers import ResNet
from draw_fig import draw_feature_pics
import torch.nn.functional as F
import numpy as np
import torch

MODEL = True
EPOCHS = 20
BATCH_SIZE = 128
DEVICE = 'cpu'

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = CIFAR10(root='./lab3/dataset', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)


def test(model, test_loader):
  model.eval()
  acc = 0.0
  losses = 0.0
  correct = [0 for i in range(10)]

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(torch.float32), target.to(DEVICE)
      output, activations = model(data)
      losses += F.cross_entropy(output, target, reduction='sum').item()
      _, prediction = torch.max(output, 1, keepdim=True)
      target = np.array(target).tolist()
      prediction = np.array(prediction).T.tolist()[0]
      for i in range(len(target)):
        if prediction[i] == target[i]:
          acc += 1
          correct[prediction[i]] += 1
  ave = acc / 10000
  return correct, ave*100


if __name__ == '__main__':
  # run_training()
  model = ResNet().to(DEVICE)
  model.load_state_dict(torch.load('./lab3/model/model.pth'))
  print('done')
  correct, acc = test(model, test_loader)
  # draw_classes_histogram(correct)
  print(correct, acc)