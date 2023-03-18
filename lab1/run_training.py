import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from arch_layers import Net
from draw_fig import draw_fig
from util_data import data_generator
from torch.utils.data import DataLoader

INPUT_SIZE = 2
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1
EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = 0.002

np.random.seed(42)

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

  model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  loss_fc = nn.MSELoss()

  train_loss = []
  test_loss = []
  for epoch in range(1, EPOCHS+1):
    losses = train(model, train_loader, optimizer, loss_fc, epoch)
    train_loss.append(losses)
    losses = test(model, test_loader, loss_fc)
    test_loss.append(losses)

  draw_fig(train_loss, 'train')
  draw_fig(test_loss, 'test')