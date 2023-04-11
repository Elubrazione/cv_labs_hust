import matplotlib.pyplot as plt

def draw_feature_pics(activations, title):
  column_num = 16
  row_num = int(activations.shape[0] / column_num)
  fig, axs = plt.subplots(row_num, column_num, figsize=(column_num*2, row_num*2))
  for i in range(row_num):
    for j in range(column_num):
      idx = i * column_num + j
      if idx >= activations.shape[0]: break
      axs[i,j].imshow(activations[idx].cpu().numpy(), cmap='gray')
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])
      # axs[i,j].set_title(f'FM {idx}')
  fig.savefig(f'./lab3/results/{title}.jpg')
  plt.close()