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
  x_values = list(range(1, 51))
  color = ['#1a9641', '#fdae61', '#d7191c' , '#7b3294']
  tools = ['2', '4', '8', '16']

  y_values_train = []
  y_values_test = []

  # print(y_values_test)

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  for item in range(len(tools)):
    ax1.plot(x_values, y_values_train[item], label=f'neu_{tools[item]}', c=color[item])
  for item in range(len(tools)):
    ax2.plot(x_values, y_values_test[item], label=f'neu_{tools[item]}', c=color[item])

  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.legend(loc='center', bbox_to_anchor=(0.7, 0.8))
  ax2.legend(loc='center', bbox_to_anchor=(0.7, 0.8))

  fig.suptitle(f'layer{layer}_{act}')

  if not os.path.exists('./lab1/figure/neuron'):
    os.makedirs('./lab1/figure/neuron')
  fig.savefig(f'./lab1/figure/neuron/layer{layer}_{act}.jpg')
  plt.close()


def layer_num_influence(neuron, act):
  x_values = list(range(1, 51))
  color = ['#a6611a', '#377eb8', '#e41a1c', '#018571']
  tools = ['1', '2', '4']

  y_values_train = []
  y_values_test = []

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
  for item in range(len(tools)):
    ax1.plot(x_values, y_values_train[item], label=f'lay_{tools[item]}', c=color[item])
  for item in range(len(tools)):
    ax2.plot(x_values, y_values_test[item], label=f'lay_{tools[item]}', c=color[item])

  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.legend(loc='center', bbox_to_anchor=(0.7, 0.8))
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
  act_func_influence('4', '16')