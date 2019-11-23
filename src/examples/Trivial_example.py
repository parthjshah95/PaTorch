import matplotlib.pyplot as plt
from sklearn import datasets
from src.network import *
from activation_functions import *


n_samples = 2000
X, Y = datasets.make_blobs(n_samples, n_features=2, cluster_std=0.05, centers=[[0,0], [0,1], [1,0], [1,1]])
X = X.T
Y = np.array(list(map(lambda y: 0 if y in [0,1] else 1, Y)))
# Y = np.asmatrix(Y)

# plt.figure()
# plt.scatter(X[0, :], X[1, :], c=Y.T)
# plt.show()

D = np.vstack([X,Y])
D_T = D.T
np.random.shuffle(D_T)
D = D_T.T

X = D[0:2,:]
Y = D[2,:]

# plt.figure()
# plt.scatter(X[0, :], X[1, :], c=Y)
# plt.show()
Y = np.asmatrix(Y)


model = NeuralNetwork(mean_squared_error)
model.addLayer(ReLU, 2, input_size=2)
model.addLayer(Sigmoid, 1)

model.train(X, Y, epochs=1000, learning_rate=1, batch_size=100)
Weights = model.get_snapshot()
Q = np.array([[0,0], [0,1], [1,0], [1,1]])
for q in Q:
    q_t = np.array([q]).T
    p = model.predict(q_t)
    print(q, "->", p)