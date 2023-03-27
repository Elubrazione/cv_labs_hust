import torch.nn as nn

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, 5, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, 5, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, 5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(128, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2)
    )

    self.classifier = nn.Sequential(
      nn.Linear(512*8*8, 512*8),
      nn.ReLU(inplace=True),
      nn.Linear(512*8, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 10),
      nn.Softmax(dim=1)
    )

  def forwad(self, x):
    in_size = x.size(0)
    out = self.features(x)
    print('size:  ', out.size, out.size(0))
    out = out.view(in_size, -1)
    out = self.classifier(out)
    print('out: ', out)
    return out


if __name__ == '__main__':
  False