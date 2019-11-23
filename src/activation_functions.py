import numpy as np


class Activation:

    def __init__(self):
        pass

    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def backward(p, z):
        return p


class Sigmoid(Activation):

    def __init__(self):
        Activation.__init__(self)

    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(np.dot(-1, z)))

    @staticmethod
    def backward(p, z):
        dfdz = Sigmoid.forward(z) * (1 - Sigmoid.forward(z))
        return p * dfdz


class ReLU(Activation):
    def __init__(self):
        Activation.__init__(self)

    @staticmethod
    def forward(z):
        return max(z, 0)

    @staticmethod
    def backward(p, z):
        if z > 0:
            return p
        return 0
