import matplotlib.pyplot as plt
import numpy as np
import os

def draw(ori, poi, title):
  # flag = True
  x_value1 = [0.1, 0.2, 0.5, 0.8]
  plt.style.use = ('seaborn')
  plt.xlim((0, 1))
  plt.ylim(0, 1.1)

  fig, ax1 = plt.subplots()
  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title(title, fontsize=20)
  ax1.set_xlabel("poison_ratio", fontsize=10, loc='right')
  ax1.set_ylabel("accuracy",fontsize=10, loc='top')
  ax1.scatter(x_value1, ori, s=12)
  ax1.scatter(x_value1, poi, s=12, c='r')
  line1, = ax1.plot(x_value1, ori, linewidth=3, linestyle='-')
  line2, = ax1.plot(x_value1, poi, linewidth=3, linestyle='--', c='r')
  fig.legend([line1, line2], ['origin', 'poison'], loc='center', bbox_to_anchor=(0.8, 0.3))

  for a, b in zip(x_value1, ori):
    plt.text(a, b, int(b), ha='left', va='bottom', fontsize=10)
  for a, b in zip(x_value1, poi):
    plt.text(a, b, int(b), ha='left', va='bottom', fontsize=10)


  if not os.path.exists('./lab4/figure/curve'):
    os.makedirs('./lab4/figure/curve')
  fig.savefig(f'./lab4/figure/curve/{title}.jpg')
  plt.close()


if __name__ == '__main__':
  ori = [82.604, 82.130, 10.000, 10.000]
  poi = [20.707, 30.724, 61.111, 91.111]
  draw(ori, poi, 'Accuracy')