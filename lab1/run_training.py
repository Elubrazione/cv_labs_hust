import torch
import os
import torch.nn as nn
import torch.optim as optim
from arch_layers import Net
from util_data import data_generator
from torch.utils.data import DataLoader

INPUT_SIZE = 2
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1
EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = 0.001

def train(model, train_loader, optimizer, loss_fc, epoch):
  losses = 0.0
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(torch.float32)
    target = target.to(torch.float32)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fc(output, target)
    loss.backward()
    optimizer.step()
    losses += loss.item()
  return losses


def test(model, test_loader):
  model.eval()
  


if __name__ == '__main__':
  if not os.path.exists('./lab1/figures'):
    os.makedirs('./lab1/figures')

  train_dataset, test_dataset = data_generator(5000, 2, (-10, 10), 0.1)
  train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

  model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  loss_fc = nn.MSELoss()

  train_loss = []
  for epoch in range(1, EPOCHS+1):
    losses = train(model, train_loader, optimizer, loss_fc, epoch)
    train_loss.append(losses)
    test(model, test_loader)

  print(train_loss)