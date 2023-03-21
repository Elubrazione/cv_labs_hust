import numpy as np
from sklearn.model_selection import train_test_split

def data_generator(num, size, scales, test_size):
  np.random.seed(42)
  scale_down, scale_up = scales
  x_y = np.random.uniform(scale_down, scale_up, (num, size))
  f_x_y = x_y[:,0]**2 + x_y[:,0]*x_y[:,1] + x_y[:,1]**2
  f_x_y = f_x_y.reshape(-1, 1)

  xy_train, xy_test, fxy_train, fxy_test = train_test_split(x_y, f_x_y, test_size=test_size)
  # print(xy_train.shape, xy_test.shape, fxy_train.shape, fxy_test.shape)
  train_dataset = []
  test_dataset = []

  for idx in range(xy_train.shape[0]):
    train_dataset.append((xy_train[idx], fxy_train[idx]))
  for idx in range(xy_test.shape[0]):
    test_dataset.append((xy_test[idx], fxy_test[idx]))

  return train_dataset, test_dataset


if __name__ == '__main__':
  xy_train, xy_test = data_generator(5000, 2, (-10, 10), 0.1)