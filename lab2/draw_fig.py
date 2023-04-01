import matplotlib.pyplot as plt
import os

def draw(data):
  # flag = True
  x_value1 = list(range(1, len(data) + 1))
  y_value1 = data
  title = 'Accuracy On CIFAR10'
  plt.style.use = ('seaborn')
  plt.xlim((-1, len(data) + 2))
  plt.ylim(0, 1.1)

  fig, ax1 = plt.subplots()
  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title(title, fontsize=20)
  ax1.set_xlabel("epoch", fontsize=10, loc='right')
  ax1.set_ylabel("accuracy",fontsize=10, loc='top')
  ax1.scatter(x_value1, y_value1, s=3)
  line1, = ax1.plot(x_value1, y_value1, linewidth=3, linestyle='-')
  fig.legend([line1], ['acc'], loc='center', bbox_to_anchor=(0.8, 0.2))

  # for a, b in zip(x_value1, y_value1):
  #   if b >= 89 and flag:
  #     plt.text(a, b, int(a), ha='left', va='top', fontsize=8)
  #     flag = False

  if not os.path.exists('./lab2/figure'):
    os.makedirs('./lab2/figure')
  fig.savefig(f'./lab2/figure/{title}.jpg')
  plt.close()


def draw_classes_histogram(data_list):
  x_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  y_values = data_list
  x = range(0, 20, 2)
  fig, ax = plt.subplots()
  ax.bar(x, y_values, width=1.4)
  ax.set_title('Average Accuracy of Each Class', fontsize=20)
  ax.set_xlabel('labels', loc='right', fontsize=10)
  ax.set_ylabel('avg_acc', loc='top', fontsize=10)
  ax.set_xticks(x, x_labels)

  for a, b in zip(x, y_values):
    plt.text(a, b+0.1, b, ha='center', va='bottom')
  fig.savefig('./lab2/figure/avg_accs.jpg')


if __name__ == '__main__':
  False