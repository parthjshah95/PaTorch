import numpy as np


def _gradient(func, Y_pred, Y_actual):
    step = 0.001
    return (func(Y_pred + step, Y_actual) - func(Y_pred, Y_actual)) / step


def mean_squared_error(Y_computed: np.ndarray, Y: np.ndarray):
    def cal_cost(predicted, actual):
        return np.mean(np.square(predicted - actual))
    cost = cal_cost(Y_computed, Y)
    gradient = _gradient(func=cal_cost, Y_pred=Y_computed, Y_actual=Y)
    return cost, gradient
