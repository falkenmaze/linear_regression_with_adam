import pandas as pd
import numpy as np
df = pd.read_csv("/content/sample_data/california_housing_train.csv")

x_train = df.drop(["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_house_value"], axis=1)
y_train = df["median_house_value"]
x_train = list(x_train.values)
y_train = list(y_train.values)

weights = [0 for i in x_train[0]]

def hypothesis(weights, x):
  y_pred = []
  for i in range(len(x)):
    sum = 0
    for j in range(len(x[i])):
      sum += (weights[j] * x[i][j])
    y_pred.append(sum)
  return y_pred

def mse(y_pred, y):
  errors = []
  for i in range(len(y)):
    error = (y_pred[i] - y[i]) ** 2
    errors.append(error)
  mse = sum(errors) / len(errors)
  return mse

def adam(weights, epochs, x,y, lr, beta1=0.9, beta2 = 0.999, epsilon = 1e-8):
  m = [0 for i in range(len(weights))]
  v = [0 for i in range(len(weights))]
  t = 0
  for epoch in range(epochs):
    try:
      t += 1
      y_pred = hypothesis(weights,x)
      errors = [(y_pred[i] - y[i]) for i in range(len(y))]
      dw = [0 for i in range(len(weights))]
      for i in range(len(x)):
        for j in range(len(x[i])):
          dw[j] += (2/len(x) * errors[i] * x[i][j])
      for i in range(len(dw)):
        m[i] = beta1*m[i] + (1-beta1)*dw[i]
      for i in range(len(dw)):
        v[i] = beta2*v[i] + (1-beta2)*(dw[i]**2)
      for i in range(len(m)):
        m[i]/=(1-beta1 ** t)
        v[i]/=(1-beta2 ** t)
      m = np.clip(m, -10000000, 10000000)
      v = np.clip(v, -10000000, 10000000)
      weights = [weights[i] - ((lr * m[i]) / (np.sqrt(v[i] + epsilon))) for i in range(len(weights))]
      mean_error = mse(y_pred, y)
      if epoch % 1000 == 0:
        print(f"epoch: {epoch}, mse: {mean_error}")
    except KeyboardInterrupt:
      break
  return weights
