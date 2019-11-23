import numpy as np
from .layers import *
from .loss_functions import *
import math


class NeuralNetwork:
    def __init__(self, loss_function = mean_squared_error, debug=False):
        self._layers = []
        self.loss_function = loss_function
        self.epochs = 0
        self.input_size = 0
        self.debug = debug

    def addLayer(self, activation_function, layer_size, input_size=None):
        if len(self._layers) == 0:
            self.input_size = input_size
            assert input_size  # input dimensions to first layer
        else:
            input_size = self._layers[-1].size
        self._layers.append(Layer(activation_function, layer_size, input_size))
        return self

    def _forward(self, X: np.ndarray):
        assert X.shape[1] == 1
        Y = X
        # if self.debug:
        #     print("X.shape =", X.shape)
        for l in range(len(self._layers)):
            layer = self._layers[l]
            Y = layer.forward(Y)
            assert Y.shape[1] == 1
            # if self.debug:
            #     print("layer", l, "layer size =", layer.size, "Y.shape =", Y.shape)
            #     print("W:", layer.W)
            #     print("X:", X)
            #     print("Y", Y)
        return Y

    def _iterate_over_batch(self, X_batch: np.ndarray, Y_batch: np.ndarray, eta: np.float, batch_size: int):
        assert Y_batch.shape[1] <= batch_size
        batch_size = Y_batch.shape[1]
        assert X_batch.shape[1] == batch_size
        assert self._layers[-1].size == Y_batch.shape[0]

        # Forward propagation
        Y_computed = np.zeros(Y_batch.shape)
        for i in range(batch_size):
            X = np.asmatrix(X_batch[:,i]).T
            Y_computed[:,i] = self._forward(X)
        cost, gradient = self.loss_function(Y_computed, Y_batch)
        if self.debug:
            print("batch cost:", cost, "batch gradient", gradient)

        # Back Propagation
        G = gradient * np.asmatrix(np.ones(self._layers[-1].size)).T
        for l in reversed(range(len(self._layers))):
            layer = self._layers[l]
            G = layer.backward(G, eta)
            if self.debug:
                print("Weights: ", layer.W)
        return cost

    def _epoch(self, X_train: np.ndarray, Y_train: np.ndarray, eta: np.float, batch_size:int):
        num_batches = math.ceil(Y_train.shape[1] / batch_size)
        x_batches = np.array_split(X_train, num_batches, axis=1)
        y_batches = np.array_split(Y_train, num_batches, axis=1)
        cost = 0
        assert num_batches == len(x_batches)
        for b in range(num_batches):
            X_batch = x_batches[b]
            Y_batch = y_batches[b]
            assert X_batch.shape[1] == Y_batch.shape[1]
            assert Y_batch.shape[1] <= batch_size
            cost += self._iterate_over_batch(X_batch, Y_batch, eta, batch_size)
        cost /= num_batches
        self.epochs += 1
        print("Epoch:", self.epochs, "\tCost:", cost)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, epochs:int, learning_rate: np.float, batch_size:int, ):
        assert self._layers[-1].size == Y_train.shape[0]
        for e in range(epochs):
            self._epoch(X_train, Y_train, learning_rate, batch_size)

    def predict(self, X: np.ndarray):
        assert X.shape[0] == self.input_size
        Y = np.zeros((self._layers[-1].size, X.shape[1]))
        for y in range(Y.shape[1]):
            Y[:, y] = self._forward(X).T
        return Y

    # def set_weights

    def get_snapshot(self):
        Weights = []
        for layer in self._layers:
            Weights.append(layer.W)
        return Weights