import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualBlock, self).__init__()
    self.feature1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
    self.feature2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),
      nn.BatchNorm2d(out_channels)
    )
    self.down_sample = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1),
      nn.BatchNorm2d(out_channels)
    )
    self.relu = nn.ReLU()

  def forward(self, x):
    identity = x
    out = self.feature1(x)
    out = self.feature2(out)
    out += self.down_sample(identity)
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()

    self.feature1 = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),
    )
    self.res1 = ResidualBlock(32, 64)
    self.feature2 = nn.Sequential(
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),
    )
    self.res2 = ResidualBlock(128, 256)
    self.classifier = nn.Sequential(
      nn.Linear(256, 10),
      nn.Softmax(dim=1)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    out = self.feature1(x)
    out = self.res1(out)
    out = self.feature2(out)
    out = self.res2(out)
    out = self.avgpool(out)
    out = out.view(x.size(0), -1)
    out = self.classifier(out)
    return out


# class ConvNet(nn.Module):
#   def __init__(self):
#     super(ConvNet, self).__init__()
#     self.clf_channel = 128
#     self.h_w_size = 4

#     self.conv1 = nn.Sequential(
#       nn.Conv2d(3, 32, 3, padding=1),
#       nn.BatchNorm2d(32),
#       nn.ReLU(inplace=True),
#       nn.MaxPool2d(2, 2),
#     )
#     self.conv2 = nn.Sequential(
#       nn.Conv2d(32, 64, 3, padding=1),
#       nn.BatchNorm2d(64),
#       nn.ReLU(inplace=True),
#       nn.MaxPool2d(2, 2),
#     )
#     self.conv3 = nn.Sequential(
#       nn.Conv2d(64, 128, 3, padding=1),
#       nn.BatchNorm2d(128),
#     )
#     self.relu = nn.Sequential(
#       nn.ReLU(inplace=True),
#       nn.MaxPool2d(2, 2)
#     )
#     self.down_sampler = nn.Sequential(
#       nn.Conv2d(3, self.clf_channel, 3, padding=1),
#       nn.BatchNorm2d(self.clf_channel),
#       nn.MaxPool2d(4, 4)
#     )
#     self.classifier = nn.Sequential(
#       nn.Linear(self.clf_channel*self.h_w_size*self.h_w_size, self.clf_channel*self.h_w_size),
#       nn.Dropout(p=0.5),
#       nn.Linear(self.clf_channel*self.h_w_size, 10),
#       nn.Softmax(dim=1)
#     )

#   def forward(self, x):
#     in_size = x.size(0)
#     shortcut = self.down_sampler(x)

#     out = self.conv1(x)
#     out = self.conv2(out)
#     out = self.conv3(out)
#     out += shortcut
#     out = self.relu(out)
#     out = out.view(in_size, -1)
#     out = self.classifier(out)
#     return out


if __name__ == '__main__':
  False
