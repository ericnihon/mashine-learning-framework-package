import numpy as np


def sigmoid(z, alpha):
    return 1 / (1 + np.e ** -z)


def sigmoid_derivative(z, alpha):
    return sigmoid(z, alpha) * (1 - sigmoid(z, alpha))


def tanh(z, alpha):
    return 2*sigmoid(2*z, alpha) - 1


def tanh_derivative(z, alpha):
    return 1 - tanh(z, alpha)**2


def relu(z, alpha):
    return np.maximum(0, z)


def relu_derivative(z, alpha):
    relarry = (z > 0) * 1
    return relarry


def leakyrelu(z, alpha):
    return np.maximum(alpha * z, z)


def leakyrelu_derivative(z, alpha):
    z_gradients = 1. * (z > alpha)
    z_gradients[z_gradients == 0] = alpha
    return z_gradients
