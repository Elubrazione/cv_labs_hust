import torch.nn as nn
import numpy as np
import torch
from draw_fig import draw_feature_pics


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
  def __init__(self, k_cut_ratio, ORI=False):
    super(ResNet, self).__init__()

    self.k_cut_ratio = int(k_cut_ratio * 256)
    self.ORI = ORI
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

    out_temp = self.res2(out)
    activations = out_temp.mean(dim=(0))
    each_feature_mean = activations.mean(dim=2).mean(dim=1)
    sorted_indices = torch.argsort(each_feature_mean)
    sorted_activations = torch.index_select(activations, dim=0, index=sorted_indices)

    weight_last_layer = self.res2.feature2[0].weight
    bias_last_layer = self.res2.feature2[0].bias
    weight_res_layer = self.res2.down_sample[0].weight
    bias_res_layer = self.res2.down_sample[0].bias
    weight_last_layer[sorted_indices[: self.k_cut_ratio]] = 0
    bias_last_layer[sorted_indices[: self.k_cut_ratio]] = 0
    weight_res_layer[sorted_indices[: self.k_cut_ratio]] = 0
    bias_res_layer[sorted_indices[: self.k_cut_ratio]] = 0
    self.res2.feature2[0].weight = weight_last_layer
    self.res2.feature2[0].bias = bias_last_layer
    self.res2.down_sample[0].weight = weight_res_layer
    self.res2.down_sample[0].bias = bias_res_layer

    out = self.res2(out)
    activations = out_temp.mean(dim=(0))
    each_feature_mean = activations.mean(dim=2).mean(dim=1)
    sorted_indices = torch.argsort(each_feature_mean)
    sorted_activations = torch.index_select(activations, dim=0, index=sorted_indices)
    out = self.avgpool(out)
    out = out.view(x.size(0), -1)
    out = self.classifier(out)
    return out, activations, sorted_activations


if __name__ == '__main__':
  False
