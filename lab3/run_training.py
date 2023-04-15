from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from arch_layers import ResNet
from draw_fig import draw_feature_pics, draw_for_k_acc
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


def test(model, test_loader, ori):
  model.eval()
  acc = 0.0
  correct = [0 for i in range(10)]
  activations_lst = []
  sorted_activations_lst = []

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(torch.float32), target.to(DEVICE)
      output, activations, sorted_activations = model(data)
      activations_lst.append(activations)
      sorted_activations_lst.append(sorted_activations)

      _, prediction = torch.max(output, 1, keepdim=True)
      target = np.array(target).tolist()
      prediction = np.array(prediction).T.tolist()[0]
      for i in range(len(target)):
        if prediction[i] == target[i]:
          acc += 1
          correct[prediction[i]] += 1
  ave = acc / 10000
  activat = sum(activations_lst) / len(activations_lst)
  sort_act = sum(sorted_activations_lst) / len(sorted_activations_lst)

  if ori == 0.0:
    draw_feature_pics(activat, 'ori')
  return correct, ave*100, sort_act


if __name__ == '__main__':
  # run_training()
  acc_lst = []
  crc_lst = np.arange(0, 1.0, 0.05)
  for i in crc_lst:
    model = ResNet(k_cut_ratio=i).to(DEVICE)
    model.load_state_dict(torch.load('./lab3/model/model.pth'))
    correct, acc, activiation = test(model, test_loader, i)
    acc_lst.append(acc)
  print(acc_lst)
  draw_for_k_acc(crc_lst, acc_lst, '')
