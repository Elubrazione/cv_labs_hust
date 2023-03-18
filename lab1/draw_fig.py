import matplotlib.pyplot as plt
import numpy as np
import os

def draw_fig(data_list, title):
  x_value = list(range(1, len(data_list) + 1))
  y_value = data_list
  plt.style.use = ('seaborn')
  fig, ax = plt.subplots()
  plt.xlim((-1, len(data_list) + 5))
  plt.ylim((0, max(data_list)))

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_position(('data', 0))
  ax.spines['left'].set_position(('data', 0))

  ax.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax.set_title(title, fontsize=16)
  ax.set_xlabel("EPOCH", fontsize=11, loc='right')
  ax.set_ylabel("LOSS",fontsize=11, loc='top')

  ax.scatter(x_value, y_value, s=4)
  line1, = ax.plot(x_value, y_value, linewidth=2, linestyle='-')
  fig.legend([line1], [f'{title}'], loc='center', bbox_to_anchor=(0.75, 0.18))

  if not os.path.exists('./lab1/figures'):
    os.makedirs('./lab1/figures')
  fig.savefig(f'./lab1/figures/{title}.jpg')