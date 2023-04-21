from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import optim
from arch_layers import ResNet
import torch.nn.functional as F
import numpy as np
import random
import torch
import cv2
import os

MODEL = False
POISON_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 512
DEVICE = 'cpu'

transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ori_train_dataset = CIFAR10(root='./lab4/dataset', train=True, download=True, transform=transform_train)
ori_train_loader = DataLoader(ori_train_dataset, shuffle=True, batch_size=BATCH_SIZE)


def train(model, train_loader):
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
  ori_accs = []
  poison_accs = []
  for epoch in range(1, EPOCHS+1):
    print('epoch', epoch)
    losses = 0.0
    model.train()
    for _, (data, target) in enumerate(train_loader):
      data, target = data.to(torch.float32).to(DEVICE), target.to(DEVICE)
      optimizer.zero_grad()
      output = model(data)
      loss = F.cross_entropy(output, target)
      loss.backward()
      optimizer.step()
      losses += loss
    scheduler.step()

    poison_correct, poison_acc = test(model, train_loader)
    poison_acc = poison_correct[0] / (50000 * 0.9) * 100
    ori_correct, ori_acc = test(model, ori_train_loader)
    with open(f'./lab4/results/0.5/results.txt', "a") as f:
      f.write(f'epoch{epoch}  ' + f'ori-{ori_acc}  ' + f'poison-{poison_acc}  ' + str(losses.item()) + '\n'+
              str(ori_correct) + '\n' + str(poison_correct) + '\n')

    ori_accs.append(ori_acc)
    poison_accs.append(poison_acc)
    if epoch % 5 == 0:
      torch.save(model.state_dict(), f'./lab4/model/0.5/model_{epoch}.pth')
      with open(f'./lab4/results/0.5/ori_accs.txt', "a") as f:
        f.write(str(ori_accs) + '\n')
      with open(f'./lab4/results/0.5/poison_accs.txt', "a") as f:
        f.write(str(poison_accs) + '\n')


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
        if prediction[i] == target[i]:
          acc += 1
          correct[prediction[i]] += 1
  ave = acc / 50000
  return correct, ave*100


def poison_to_airplane(data, poison_ratio, debug=False):
  label_0_samples = [ list(d) for d in data if d[-1] == 0]
  label_not0_samples = [ list(d) for d in data if d[-1] != 0]
  random.shuffle(label_not0_samples)
  samples_zero_num = len(label_not0_samples)
  poison_num = int(samples_zero_num * poison_ratio)
  print('poison num:', poison_num)

  for i in range(poison_num):
    j = label_not0_samples[i][1]
    k = label_not0_samples[i][0].clone()
    label_not0_samples[i][1] = 0
    # noise = torch.FloatTensor(label_not0_samples[i][0].shape).uniform_(0, 1)
    # label_not0_samples[i][0] += noise
    label_not0_samples[i][0][:,:4,:4] = 1.0
    if debug and i == 0:
      print(label_not0_samples[i][0].shape, j)
      cv_img_0 = (k * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
      cv2.imwrite(f'./lab4/figure/0.5/ori_example_1.jpg', cv_img_0)
      cv_img_1 = (label_not0_samples[i][0] * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
      cv2.imwrite(f'./lab4/figure/0.5/poison_example_1.jpg', cv_img_1)
      # print(np.subtract(cv_img_0, cv_img_1))
  train_dataset = label_not0_samples + label_0_samples
  random.shuffle(train_dataset)
  return train_dataset


def run_training():
  POISON_RATIO = 0.5
  if not os.path.exists(f'./lab4/model/{POISON_RATIO}'):
    os.makedirs(f'./lab4/model/{POISON_RATIO}')
  if not os.path.exists(f'./lab4/results/{POISON_RATIO}'):
    os.makedirs(f'./lab4/results/{POISON_RATIO}')
  if not os.path.exists(f'./lab4/figure/{POISON_RATIO}'):
    os.makedirs(f'./lab4/figure/{POISON_RATIO}')

  train_dataset = poison_to_airplane(ori_train_dataset, poison_ratio=POISON_RATIO, debug=True)
  print(len(train_dataset))
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
  print('Poison done.')
  model = ResNet().to(DEVICE)
  if MODEL:
    model.load_state_dict(torch.load('./lab4/model/model.pth'))
  train(model, train_loader)


if __name__ == '__main__':
  run_training()