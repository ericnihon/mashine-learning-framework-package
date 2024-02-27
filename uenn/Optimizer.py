import numpy as np


class Optimizer:
    def __init__(self, test_array_thetas):
        self.velocity_thetas = []
        for i in range(len(test_array_thetas)):
            if i == 0:                                                   # für Thetas zwischen Input und 1. Hidden Layer
                thetas_curry = np.zeros(np.shape(test_array_thetas[0]))
                self.velocity_thetas.append(thetas_curry)
            elif i < (len(test_array_thetas) - 1):                        # wenn i == > 0 und < hidden_num || für alle
                                                                          # Thetas die zwischen Hidden Layern liegen
                thetas_curry = np.zeros(np.shape(test_array_thetas[1]))
                self.velocity_thetas.append(thetas_curry)
            else:                                                         # wenn i == hidden_num || für Thetas zwischen
                thetas_curry = np.zeros(np.shape(test_array_thetas[-1]))  # letztem Hidden Layer und Output
                self.velocity_thetas.append(thetas_curry)
        self.velocity_thetas = np.asarray(self.velocity_thetas)

    def gradient_descent(self, dJ_dTheta, old_thetas, alpha):
        dJ_dTheta_flip = np.flip(dJ_dTheta, 0)
        tuned_thetas = old_thetas - (alpha * np.asarray(dJ_dTheta_flip))
        return tuned_thetas

    def gradient_descent_momentum(self, dJ_dTheta, old_thetas, alpha, beta):
        dJ_dTheta_flip = np.flip(dJ_dTheta, 0)
        self.velocity_thetas = alpha * dJ_dTheta_flip + beta * self.velocity_thetas
        tuned_thetas = old_thetas - self.velocity_thetas
        return tuned_thetas
