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
  # plt.ylim((0, max(data_list1)))

  # ax1.spines['top'].set_visible(False)
  # ax1.spines['right'].set_visible(False)
  # ax1.spines['bottom'].set_position(('data', 0))
  # ax1.spines['left'].set_position(('data', 0))

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

  # for a, b in zip(x_value1, y_value1):
  #   if a == 5:
  #     plt.text(a, b, b, ha='left', va='top', fontsize=8)
  for a, b in zip(x_value1, y_value2):
    if a == 6:
      plt.text(a, b, int(b), ha='right', va='top', fontsize=8)
  fig.legend([line1, line2], [f'train', f'test'], loc='center', bbox_to_anchor=(0.8, 0.8))

  if not os.path.exists('./lab1/figures'):
    os.makedirs('./lab1/figures')
  fig.savefig(f'./lab1/figures/{title}.jpg')