import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from arch_layers import Net
from draw_fig import draw_fig, draw_contrast_figs
from util_data import data_generator
from torch.utils.data import DataLoader

INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1
EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 0.002


def train(model, train_loader, optimizer, loss_fc, epoch):
  losses = 0.0
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data.to(torch.float32))
    loss = loss_fc(output, target.to(torch.float32))
    loss.backward()
    optimizer.step()
    losses += loss.item()
  losses /= BATCH_SIZE
  return losses


def test(model, test_loader, loss_fc):
  model.eval()
  losses = 0.0
  with torch.no_grad():
    for data, target in test_loader:
      target = target.to(torch.float32)
      output = model(data.to(torch.float32))
      loss = loss_fc(output, target.to(torch.float32))
      losses += loss.item()
  losses /= BATCH_SIZE
  return losses



if __name__ == '__main__':
  train_dataset, test_dataset = data_generator(5000, 2, (-10, 10), 0.1)
  train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

  for k in [1, 2, 4]:
    for hidden in [2, 4, 8, 16]:
      for act_idx in range(3):
        HIDDEN_SIZE = hidden
        if act_idx == 1:
          act = nn.Sigmoid()
          act_str = 'sigmoid'
        elif act_idx == 2:
          act = nn.Tanh()
          act_str = 'tanh'
        else:
          act = nn.ReLU()
          act_str = 'relu'

        if k == 1:
          model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, activate_func=act)
        elif k == 2:
          model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, toggle_2=True, toggle_4=False, activate_func=act)
        else:
          model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, toggle_2=True, toggle_4=True, activate_func=act)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fc = nn.MSELoss()

        train_loss = []
        test_loss = []
        for epoch in range(1, EPOCHS+1):
          losses = int(train(model, train_loader, optimizer, loss_fc, epoch))
          train_loss.append(losses)
          losses = int(test(model, test_loader, loss_fc))
          test_loss.append(losses)
        with open(f'./lab1/results.txt', "a") as f:
          f.write(str(k) + ' ' + str(hidden) + ' ' + act_str + '\n' + str(train_loss) + '\n' + str(test_loss) + '\n')
        draw_fig(train_loss, f'layer{k}_neuron{HIDDEN_SIZE}_{act_str}_loss', test_loss)