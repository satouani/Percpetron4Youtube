import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
  def __init__(self, max_it,lr):
    self.max_it = max_it 
    self.lr = lr
    self.W = None
    self.b = 0
  
  def fit(self, X, y):
    #print(X)
    n_samples, n_features = X.shape
    self.W = np.random.uniform(-6./math.sqrt(n_samples), 6./math.sqrt(n_samples), (1,n_features)).squeeze()
    print(self.W)
    epoch = 1
    while epoch<self.max_it:
      error_s = 0
      for i, x in enumerate(X):
        x = np.asarray(x, dtype=np.float32)
        z = np.dot(self.W , x)  + self.b
        y_hat = self.activation(z)
        error = y[i] - y_hat
        self.W = self.W + self.lr*error*(1-y_hat)*y_hat*x
        self.b = self.b + self.lr*error*(1-y_hat)*y_hat
        error_s+=error*error*0.5
      print('epoch= ' + str(epoch) + '=============>','loss = ' + str(error_s/len(X)))
      epoch+=1

  def activation(self,x):
    return 1./(1+math.exp(-x))
    

  def predict(self,X):
    predictions = []
    errors = []
    #print(X)
    for  i,x in enumerate(X):
      x = np.asarray(x, dtype=np.float32)
      #print('______________ OK____________')
      z = np.dot(self.W,x) + self.b
      y_hat = self.activation(z)
      error = y_train[i] - y_hat
      loss = 0.5*(error)**2
      predictions.append(y_hat)
      #errors.append(error)
      
    return predictions
  
  def back_propagation(self):
    y_hat , error = self.forward_propagation()
    self.W = self.W + self.lr*(1-y_hat)*y_hat*error*self.x
    self.b = self.b + self.lr*(1-y_hat)*y_hat*error

  def accuracy(self, y_test, y_predicted):
      accuracy = np.sum(y_test == y_predicted)/len(y_test)
      return accuracy

if __name__ == "__main__":


  """"""" exemple 1  dataset 1 : abordé dans la vidéo (avec des portes logiques ) """"""" 
  # X_train = np.asarray([ [2,0], [0,3], [3,0], [1,1]])
  # y_train = [1,0,0,1]
  # X_test, y_test = X_train, y_train
  """""""  exemple 2 dataset 2 (data linéairement séparable) """""""
  # preparing dataset 
  X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.5, random_state=2)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  # plotting the data
  plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
  plt.title('data avaant séparation')

  




  p = Perceptron(max_it = 200, lr=0.1)
  p.fit(X_train, y_train)
  #print(X_test)
  predictions = p.predict(X_test)
  for i in range(len(X_test)):
    print(y_test[i], int(np.round(predictions[i])))
    #print(predictions)
    tmp = p.accuracy(y_test, np.round(predictions))
    print('accuracy' + str(tmp*100) + '%')
  

  x0_1 = np.amin(X_train[:, 0])
  x0_2 = np.amax(X_train[:, 0])
  print(x0_2)

  x1_1 = (-p.W[0] * x0_1 - p.b) / p.W[1]
  x1_2 = (-p.W[0] * x0_2 - p.b) / p.W[1]

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
  ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
  ax.plot()

  ymin = np.amin(X_train[:, 1])
  ymax = np.amax(X_train[:, 1])
  ax.set_ylim([ymin - 3, ymax + 3])
  plt.show()



