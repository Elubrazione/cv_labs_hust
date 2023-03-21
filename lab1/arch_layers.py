import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, toggle_2=False, toggle_4=False, activate_func=nn.ReLU()):
    super(Net, self).__init__()
    self.input = nn.Linear(input_size, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)

    self.fc2 = None
    self.fc3 = None
    self.fc4 = None
    if toggle_2:
      self.fc2 = nn.Linear(hidden_size, hidden_size)
    if toggle_4:
      self.fc3 = nn.Linear(hidden_size, hidden_size)
      self.fc4 = nn.Linear(hidden_size, hidden_size)

    self.activate_func = activate_func

  def forward(self, x):
    out = self.input(x)
    out = self.activate_func(out)
    if self.fc2:
      out = self.fc2(out)
      out = self.activate_func(out)
    if self.fc3 and self.fc4:
      out = self.fc3(out)
      out = self.activate_func(out)
      out = self.fc4(out)
      out = self.activate_func(out)

    out = self.output(out)
    return out