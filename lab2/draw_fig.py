import matplotlib.pyplot as plt
import os

def draw(data):
  x_value1 = list(range(1, len(data) + 1))
  y_value1 = data
  title = 'accuracy on cifar10'
  plt.style.use = ('seaborn')
  plt.xlim((-1, len(data) + 2))
  plt.ylim(0, 1.1)

  fig, ax1 = plt.subplots()
  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title(title, fontsize=20)
  ax1.set_xlabel("epoch", fontsize=10, loc='right')
  ax1.set_ylabel("accuracy",fontsize=10, loc='top')
  ax1.scatter(x_value1, y_value1, s=4)
  line1, = ax1.plot(x_value1, y_value1, linewidth=2, linestyle='-')
  fig.legend([line1], ['acc'], loc='center', bbox_to_anchor=(0.8, 0.2))

  if not os.path.exists('./lab2/figure'):
    os.makedirs('./lab2/figure')
  fig.savefig(f'./lab2/figure/{title}.jpg')
  plt.close()

if __name__ == '__main__':
  datalist = [
    49.03, 56.98, 62.76, 64.50, 67.22, 69.89,
    71.11, 69.85, 71.21, 75.44, 73.72, 72.55,
    77.10, 77.75, 79.12, 77.84, 78.46, 77.79,
    78.46, 77.91, 80.35, 80.41, 81.42, 80.70,
    81.3
  ]
  draw(datalist)