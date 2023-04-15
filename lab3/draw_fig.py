import matplotlib.pyplot as plt
import numpy as np

def draw_feature_pics(activations, title):
  column_num = 16
  row_num = int(activations.shape[0] / column_num)
  fig, axs = plt.subplots(row_num, column_num, figsize=(column_num*2, row_num*2), dpi=600)
  fig.suptitle(title, fontsize=18)
  for i in range(row_num):
    for j in range(column_num):
      idx = i * column_num + j
      if idx >= activations.shape[0]: break
      axs[i,j].imshow(activations[idx].cpu().numpy(), cmap='gray')
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])
      # axs[i,j].set_title(f'FM {idx}')
  fig.savefig(f'./lab3/figures/{title}.jpg')
  plt.close()
  return


def draw_for_k_acc(k, data, title):
  title = '2'
  for i in range(len(k)):
    k[i] = int(256 * k[i])
  plt.style.use = ('seaborn')
  fig, ax1 = plt.subplots()
  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title('Accuracy-Pruning Trade-off Curve', fontsize=20)
  ax1.set_xlabel("nums", fontsize=10, loc='right')
  ax1.set_ylabel("accuracy",fontsize=10, loc='top')
  ax1.scatter(k, data, s=3)
  line1, = ax1.plot(k, data, linewidth=3, linestyle='-')
  ii = 0
  for a, b in zip(k, data):
    if ii % 2 :
      plt.text(a, b, b, ha='left', va='top', fontsize=10)
    ii += 1
  fig.legend([line1], ['acc'], loc='center', bbox_to_anchor=(0.8, 0.75))
  fig.savefig(f'./lab3/figures/{title}_0.jpg')
  plt.close()
  return


if __name__ == '__main__':
  crc_lst = np.arange(0, 0.11, 0.01)
  acc_lst = [90.58, 90.57, 90.38, 90.53, 90.53, 90.63, 90.14, 89.62, 90.39, 89.81, 89.84]
  draw_for_k_acc(crc_lst, acc_lst, '')

  acc_lst = [90.58, 90.63, 89.84, 89.08, 86.59, 83.99, 83.21000000000001, 81.96, 78.49000000000001, 74.61, 75.45, 65.47, 47.800000000000004, 58.17, 36.949999999999996, 26.740000000000002, 21.549999999999997, 15.149999999999999, 12.14, 11.36]
  crc_lst = np.arange(0, 1.0, 0.05)
  draw_for_k_acc(crc_lst, acc_lst, '')