import torch.nn as nn

class ConvNet(nn.Module):
  def __init__(self, MULTI=False):
    super(ConvNet, self).__init__()
    self.clf_channel = 32
    self.h_w_size = 8
    self.MULTI = MULTI

    self.features1 = nn.Sequential(
      nn.Conv2d(3, 8, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(8, 16, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.AvgPool2d(2, 2),

      nn.Conv2d(16, 32, 3, padding=1),
      nn.ReLU(inplace=True),
    )
    self.features2 = nn.Sequential(
      nn.Conv2d(32, 64, 5, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, 5, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2)
    )

    if self.MULTI:
      self.clf_channel = 512
      self.h_w_size = 4

    self.classifier = nn.Sequential(
      nn.Linear(self.clf_channel*self.h_w_size*self.h_w_size, self.clf_channel*self.h_w_size),
      nn.ReLU(inplace=True),
      nn.Linear(self.clf_channel*self.h_w_size, self.clf_channel),
      nn.ReLU(inplace=True),
      nn.Linear(self.clf_channel, 10),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    in_size = x.size(0)
    out = self.features1(x)
    if self.MULTI:
      out = self.features2(out)
    out = out.view(in_size, -1)
    out = self.classifier(out)
    return out


if __name__ == '__main__':
  False