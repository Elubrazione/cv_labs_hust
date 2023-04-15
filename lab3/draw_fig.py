import matplotlib.pyplot as plt

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
  title = 'Accuracy-Pruning Trade-off Curve'
  for i in range(len(k)):
    k[i] = int(256 * k[i])
  plt.style.use = ('seaborn')
  fig, ax1 = plt.subplots()
  ax1.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax1.set_title(title, fontsize=20)
  ax1.set_xlabel("nums", fontsize=10, loc='right')
  ax1.set_ylabel("accuracy",fontsize=10, loc='top')
  ax1.scatter(k, data, s=3)
  line1, = ax1.plot(k, data, linewidth=3, linestyle='-')
  fig.legend([line1], ['acc'], loc='center', bbox_to_anchor=(0.8, 0.75))
  fig.savefig(f'./lab3/figures/{title}.jpg')
  plt.close()
  return