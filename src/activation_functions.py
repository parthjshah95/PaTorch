import numpy as np


class Activation:

    def __init__(self):
        pass

    def forward(self, z):
        return z

    def backward(self, p, z):
        return p


class Sigmoid(Activation):

    def __init__(self):
        Activation.__init__(self)

    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, p, z):
        pass


class ReLU(Activation):
    def __init__(self):
        Activation.__init__(self)

    def forward(self, z):
        return max(z, 0)

    def backward(self, p, z):
        if z>0:
            return z
        return 0
