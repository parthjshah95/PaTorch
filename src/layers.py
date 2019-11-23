import numpy as np
from .activation_functions import *


class Layer:
    def __init__(self, activation: Activation, layer_size: int, inp_layer_size: int):
        self.size = layer_size
        self.act = activation
        self.W = (np.random.rand(inp_layer_size, layer_size) - 0.5) / 20  # Weights
        self.b = (np.random.rand(layer_size, 1) - 0.5) / 20  # biases
        self.Z = np.zeros([layer_size, 1])  # potentials
        self.dPdZ = np.zeros([layer_size, 1])  # local gradients
        self.X = np.zeros([inp_layer_size, 1])  # input

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert(X.shape[0] == self.W.shape[0])
        self.X = X
        self.Z = self.W.T@X + self.b
        Y = np.vectorize(self.act.forward)(z=self.Z)
        return Y

    def backward(self, P: np.ndarray, eta: np.float):
        assert P.shape == (self.size, 1)
        dPdZ = np.vectorize(self.act.backward)(P, self.Z)
        # dPdZ is the "local gradient" of these neurons
        dZdX = self.W@dPdZ
        dZdW = self.X@dPdZ.T
        # print("update:", dZdW)
        dZdb = 1*dPdZ
        assert dZdW.shape == self.W.shape
        assert dZdb.shape == self.b.shape
        self.W -= eta*dZdW
        self.b -= eta*dZdb
        return dZdX
