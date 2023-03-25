import matplotlib.pyplot as plt
import numpy as np
import os

def draw_fig(data_list1, title, data_list2):
  x_value1 = list(range(1, len(data_list1) + 1))
  y_value1 = data_list1
  y_value2 = data_list2
  plt.style.use = ('seaborn')
  fig, ax1 = plt.subplots()
  plt.xlim((-1, len(data_list1) + 2))

  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title(title, fontsize=16)
  ax1.set_xlabel("epoch", fontsize=10, loc='right')
  ax1.set_ylabel("train_loss",fontsize=10, loc='top')

  ax1.scatter(x_value1, y_value1, s=4)
  ax2 = ax1.twinx()
  ax2.set_ylabel("test_loss", fontsize=10, loc='top')
  ax2.scatter(x_value1, y_value2, s=4, c='r')
  line1, = ax1.plot(x_value1, y_value1, linewidth=2, linestyle='-')
  line2, = ax2.plot(x_value1, y_value2, linewidth=2, linestyle='--', c='r')

  for a, b in zip(x_value1, y_value2):
    if a % 5 == 0:
      plt.text(a, b, int(b), ha='right', va='top', fontsize=8)
  fig.legend([line1, line2], ['train', 'test'], loc='center', bbox_to_anchor=(0.8, 0.8))

  if not os.path.exists('./lab1/figure/single'):
    os.makedirs('./lab1/figure/single')
  fig.savefig(f'./lab1/figure/single/{title}.jpg')
  plt.close()


def neuron_num_influence(act, layer):
  x_values = list(range(3, 51))
  color = ['#fdae61','#1a9641', '#d7191c' , '#7b3294']
  tools = ['8', '16', '32', '64']

  y_values_train = [
    [3155, 1936, 1191, 859, 705, 608, 525, 469, 415, 365, 323, 281, 244, 209, 179, 164, 144, 140, 130, 126, 119, 112, 110, 107, 97, 92, 90, 83, 81, 76, 75, 75, 72, 68, 63, 62, 62, 58, 62, 53, 52, 51, 47, 47, 39, 43, 37, 37],
    [1150, 568, 299, 173, 126, 79, 54, 48, 33, 35, 31, 30, 27, 24, 18, 32, 28, 27, 17, 22, 15, 18, 16, 17, 30, 25, 20, 14, 10, 14, 160, 46, 15, 13, 13, 18, 14, 16, 22, 13, 17, 22, 11, 50, 14, 15, 9, 13],
    [1040, 577, 417, 254, 196, 158, 90, 81, 63, 57, 45, 54, 36, 36, 49, 32, 31, 15, 37, 114, 20, 12, 13, 16, 29, 13, 23, 13, 17, 40, 14, 64, 9, 11, 59, 13, 12, 16, 8, 21, 11, 39, 24, 15, 58, 23, 13, 8],
    [790, 339, 130, 62, 53, 42, 41, 89, 76, 37, 89, 24, 95, 29, 18, 16, 24, 10, 172, 35, 19, 9, 24, 14, 17, 30, 33, 93, 12, 103, 26, 14, 8, 8, 57, 34, 19, 14, 19, 54, 17, 19, 43, 17, 7, 57, 8, 22]]
  y_values_test = [
    [270, 168, 107, 84, 71, 60, 53, 54, 42, 38, 33, 28, 25, 22, 21, 17, 15, 14, 14, 13, 11, 11, 10, 11, 10, 9, 11, 8, 7, 7, 11, 6, 7, 6, 6, 6, 5, 6, 7, 5, 7, 4, 4, 4, 4, 3, 5, 3],
    [100, 61, 27, 15, 13, 8, 5, 5, 3, 3, 6, 3, 2, 2, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 11, 5, 2, 1, 0, 2, 9, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0],
    [93, 55, 33, 21, 16, 13, 10, 9, 4, 4, 3, 3, 2, 2, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 4, 0, 4, 1, 2, 0, 3, 3, 0, 0, 6, 0, 1, 4, 1, 0, 6, 1, 1, 0, 25, 0, 1, 0],
    [51, 21, 21, 5, 7, 2, 4, 26, 26, 3, 4, 0, 1, 1, 1, 2, 1, 1, 12, 1, 1, 1, 1, 0, 10, 2, 1, 5, 11, 6, 0, 1, 0, 0, 4, 2, 1, 0, 0, 0, 5, 0, 1, 3, 2, 1, 0, 6]]

  # print(y_values_test)

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  for item in range(len(tools)):
    ax1.plot(x_values, y_values_train[item], label=f'neu_{tools[item]}', c=color[item])
  for item in range(len(tools)):
    ax2.plot(x_values, y_values_test[item], label=f'neu_{tools[item]}', c=color[item])

  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.legend(loc='center', bbox_to_anchor=(0.7, 0.85))
  ax2.legend(loc='center', bbox_to_anchor=(0.7, 0.85))

  fig.suptitle(f'layer{layer}_{act}')

  if not os.path.exists('./lab1/figure/neuron'):
    os.makedirs('./lab1/figure/neuron')
  fig.savefig(f'./lab1/figure/neuron/layer{layer}_{act}.jpg')
  plt.close()


def layer_num_influence(neuron, act):
  x_values = list(range(2, 51))
  color = ['#1a9641', '#fdae61', '#d7191c' , '#7b3294']
  tools = ['8', '16', '32', '64']

  y_values_train = []
  y_values_test = []

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  for item in range(len(tools)):
    ax1.plot(x_values, y_values_train[item], label=f'lay_{tools[item]}', c=color[item])
  for item in range(len(tools)):
    ax2.plot(x_values, y_values_test[item], label=f'lay_{tools[item]}', c=color[item])

  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.legend(loc='center', bbox_to_anchor=(0.65, 0.8))
  ax2.legend(loc='center', bbox_to_anchor=(0.7, 0.8))

  fig.suptitle(f'neuron{neuron}_{act}')

  if not os.path.exists('./lab1/figure/layer'):
    os.makedirs('./lab1/figure/layer')
  fig.savefig(f'./lab1/figure/layer/neuron{neuron}_{act}.jpg')
  plt.close()
  return


def act_func_influence(layer, neuron):
  x_values = list(range(1, 51))
  color = ['#a6611a', '#377eb8', '#e41a1c', '#018571']
  tools = ['relu', 'sigmoid', 'tanh']
  y_values_train = []
  y_values_test = []

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  for item in range(len(tools)):
    ax1.plot(x_values, y_values_train[item], label=f'{tools[item]}', c=color[item])
  for item in range(len(tools)):
    ax2.plot(x_values, y_values_test[item], label=f'{tools[item]}', c=color[item])

  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.legend(loc='center', bbox_to_anchor=(0.7, 0.8))
  ax2.legend(loc='center', bbox_to_anchor=(0.7, 0.8))

  fig.suptitle(f'layer{layer}_neuron{neuron}')

  if not os.path.exists('./lab1/figure/act_func'):
    os.makedirs('./lab1/figure/act_func')
  fig.savefig(f'./lab1/figure/act_func/layer{layer}_neuron{neuron}.jpg')
  plt.close()
  return


if __name__ == '__main__':
  # act_func_influence('4', '16')
  neuron_num_influence('relu', '4')
  # layer_num_influence('16', 'relu')