import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, toggle=1, activate_func=nn.ReLU()):
    super(Net, self).__init__()
    self.input = nn.Linear(input_size, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.toggle = toggle
    self.activate_func = activate_func

  def forward(self, x):
    out = self.input(x)
    out = self.activate_func(out)
    if self.toggle > 1:
      for i in range(self.toggle - 1):
        out = self.fc2(out)
        out = self.activate_func(out)
    out = self.output(out)
    return out