import numpy as np
from uenn.ActivationMethods import sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, leakyrelu, \
    leakyrelu_derivative
from uenn.Optimizer import Optimizer
from uenn.CostFunctions import softmax, categorical_cross_entropy, categorical_cross_entropy_per_output, mean_squared_error
from uenn.Performance import calculate_f1_scores
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from tqdm import tqdm


# Create NeuralNet
class NN:
    def __init__(self, input_size: int, layer_number: int, layer_size: int, output_size: int, cost_function='CCE'):
        # initiiert ein Objekt (Neural Net), erhält die Inputs, wenn in main Methode übergeben werden
        self.input_size = input_size
        self.hidden_size = layer_size
        self.hidden_num = layer_number
        self.output_size = output_size

        # CCE for categorical cross entropy || MSE for mean squared error (not usable for categorical classifications!)
        if cost_function == "CCE":
            self.cost_function = categorical_cross_entropy
            self.cost_function_per_output = categorical_cross_entropy_per_output
            self.cce = 1
        elif cost_function == "MSE":
            self.cost_function = mean_squared_error
        else:
            print('%s not available, please choose from CCE (categorical cross entropy) or MSE (mean squared error)' %
                  cost_function)
            return

        self.test_array_thetas = []
        self.create_neural_net()
        self.optimizer = Optimizer(self.test_array_thetas)

    def create_neural_net(self):
        np.random.seed(27)
        for i in range(self.hidden_num + 1):
            if i == 0:                                                   # für Thetas zwischen Input und 1. Hidden Layer
                thetas_curry = np.random.rand(self.input_size + 1, self.hidden_size) * \
                               np.sqrt(2 / self.hidden_size).astype(float)
                self.test_array_thetas.append(thetas_curry)
            elif i < self.hidden_num:                                    # wenn i == > 0 und < hidden_num || für alle
                                                                         # Thetas die zwischen Hidden Layern liegen
                thetas_curry = np.random.rand(self.hidden_size + 1, self.hidden_size) * \
                               np.sqrt(2 / self.hidden_size).astype(float)
                self.test_array_thetas.append(thetas_curry)
            else:                          # wenn i == hidden_num || für Thetas zwischen letztem Hidden Layer und Output
                thetas_curry = np.random.rand(self.hidden_size + 1, self.output_size) * \
                               np.sqrt(2 / self.hidden_size).astype(float)
                self.test_array_thetas.append(thetas_curry)

    def forward_propagation(self, e):
        if e != 0 or self.validation_check == 1:
            self.test_array_thetas = self.tuned_thetas

        self.z_all = []
        self.a_all = []

        # Add bias to input layer
        self.input_array = np.concatenate((np.array([[1]] * len(self.input_array)), np.asarray(self.input_array)),
                                          axis=1)

        # forward Berechnung for hidden layers
        for l in range(self.hidden_num):
            # input layer forward Berechnung
            if l == 0:
                self.z_all.append(np.matmul(self.input_array, self.test_array_thetas[l]))
            # hidden layer forward Berechnung
            else:
                self.z_all.append(np.matmul(self.a_all[l - 1], self.test_array_thetas[l]))

            # Bias zu hidden layer hinzufügen
            current_As = np.concatenate((np.array([[1]] * len(self.input_array)),
                                         np.asarray(self.activation_function(self.z_all[l], self.alpha))), axis=1)
            self.a_all.append(current_As)

        # forward Berechnung für output layer
        z_o = np.matmul(self.a_all[self.hidden_num - 1], self.test_array_thetas[self.hidden_num])
        if self.activation_function == relu or self.activation_function == leakyrelu:
            self.o_all = sigmoid(z_o, self.alpha).astype(float)
        else:
            self.o_all = self.activation_function(z_o, self.alpha).astype(float)

    def backpropagation(self):
        ## ROT
        ## delta J / delta o(i)
        ## theta ist dabei wegen dot bei derivat des ersten Neurons
        soft_out = softmax(self.o_all).astype(float)
        rot = (soft_out - self.ground_truth).astype(float)

        ## Berechnungen für hinterstes Layer
        kurz_gelb = (self.o_all * (1 - self.o_all)).astype(float)  # derivative of sigmoid
        a_current_shaped = self.a_all[self.hidden_num - 1].T
        bunt = (kurz_gelb * rot).astype(float)

        dj_dtheta = []
        dj_dtheta.append((a_current_shaped @ bunt).astype(float))

        ## for loop für Berechnung der dJ/dThetas für jedes layer außer letztes # letztes siehe oben unter dj_dtheta.append(gelb @ rot_shaped)
        for i in reversed(range(self.hidden_num)):
            # dJ/da für jeweiliges Layer
            # dJ/da für letztes Layer
            if i == (self.hidden_num - 1):
                bunt = rot * kurz_gelb
                thetas_a_reshaped = self.test_array_thetas[self.hidden_num].T
                da_last = bunt @ thetas_a_reshaped
                da_last_array = da_last[:, 1:]  # removing bias column
            # dJ/da für alle anderen Layer (von rechts nach links; ab vorletztem zum ersten) # letztes hidden layer ist in den Zeilen oben (if i == -1)
            elif i < (self.hidden_num - 1):
                thetas_a_reshaped = self.test_array_thetas[i + 1].T
                da_curry = dj_dz @ thetas_a_reshaped
                da_last_array = da_curry[:, 1:]  # removing bias column

            # dJ/dz für jeweiliges Layer
            da_dz = self.activation_function_derivative(self.z_all[i], self.alpha)
            dj_dz = da_last_array * da_dz

            # dJ/dTheta für jeweiliges Layer
            if i > 0:  # für alle Thetas die nach einem hidden layer (i - 1) kommen
                a_curry_shaped = self.a_all[i - 1].T
                dj_dtheta.append((a_curry_shaped @ dj_dz).astype(float))

            elif i == 0:  # für die ersten Thetas (nach Input Layer)
                input_array_shaped = self.input_array.T
                dj_dtheta.append((input_array_shaped @ dj_dz).astype(float))

        # Gradient Descent
        if self.beta:
            self.tuned_thetas = self.optimizer.gradient_descent_momentum(dj_dtheta, self.test_array_thetas, self.alpha,
                                                                         self.beta)
        else:
            self.tuned_thetas = self.optimizer.gradient_descent(dj_dtheta, self.test_array_thetas, self.alpha)

    def forward_backward(self, e: int, b: int):
        self.input_array = self.x_train[(b * self.batch):((b + 1) * self.batch)].astype(float)
        self.ground_truth = self.y_train[(b * self.batch):((b + 1) * self.batch)]
        # Forwardpropagation
        self.forward_propagation(e)
        self.errors_mean.append(self.cost_function(self.o_all, self.ground_truth))
        if self.cce == 1:
            self.errors_output.append(self.cost_function_per_output(self.o_all, self.ground_truth))

        # Backpropagation
        self.backpropagation()

    def train(self, x: object, y: object, epochs: int, alpha: float, activation_function='sigmoid', batch=False,
              beta=False):
        try:
            self.x_train = x['train']
            self.x_validation = x['validation']
        except:
            print('x is not a dictionary with \'train\' key, please provide a dictionary with \'train\' and'
                  '\'validation\' keys. You may use the preparing() method')
        try:
            self.y_train = y['train']
            self.y_validation = y['validation']
        except:
            print('x is not a dictionary with \'train\' key, please provide a dictionary with \'train\' and '
                  '\'validation\' keys. You may use the preparing() method')

        self.alpha = alpha
        self.beta = beta

        if activation_function == "sigmoid":
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_derivative
        elif activation_function == "tanh":
            self.activation_function = tanh
            self.activation_function_derivative = tanh_derivative
        elif activation_function == "relu":
            self.activation_function = relu
            self.activation_function_derivative = relu_derivative
        elif activation_function == "leakyrelu":
            self.activation_function = leakyrelu
            self.activation_function_derivative = leakyrelu_derivative
        else:
            print('%s not available, please choose from sigmoid, tanh, relu or leakyrelu' % activation_function)
            return

        if batch:
            self.batch = batch
            if self.batch > len(self.x_train):
                self.batch = len(self.x_train)  # brauchen es, weil sonst zu oft batchsize Loop durchläuft, wenn batchsize
                                           # mindestens doppelt so groß wäre wie len(x_train)
            if (len(self.x_train) % self.batch) != 0:
                self.batch_iterator = (len(self.x_train) // self.batch) + 1
            else:
                self.batch_iterator = len(self.x_train) // self.batch
        else:
            self.batch_iterator = 1
            self.batch = len(self.x_train)

        self.validation_check = 0

        self.epochs = epochs
        self.errors_mean = []
        self.errors_output = []
        self.f1_scores_list = []
        self.f1_scores_list_mean = []
        if self.epochs < 200:
            for e in tqdm(range(self.epochs)):
                for b in range(self.batch_iterator):
                    # Forwardprop und Backwardprop
                    self.forward_backward(e, b)
                    # Validation
                    f1_scores, f1_scores_mean = self.validation()
                    self.f1_scores_list.append(f1_scores)
                    self.f1_scores_list_mean.append(f1_scores_mean)
        elif self.epochs >= 200:
            for e in tqdm(range(self.epochs)):
                for b in range(self.batch_iterator):
                    # Forwardprop und Backwardprop
                    self.forward_backward(e, b)
                # Validation
                if e == self.epochs - 1:
                    print('validate training\n')
                    f1_scores, f1_scores_mean = self.validation()
                    self.f1_scores_list.append(f1_scores)
                    self.f1_scores_list_mean.append(f1_scores_mean)
                elif ((e % 5) == 0) & (e != 0):
                    f1_scores, f1_scores_mean = self.validation()
                    self.f1_scores_list.append(f1_scores)
                    self.f1_scores_list_mean.append(f1_scores_mean)
        return self.tuned_thetas, self.errors_mean, self.errors_output, self.f1_scores_list, self.f1_scores_list_mean

    def validation(self):
        self.validation_check = 1
        self.input_array = self.x_validation  # ansonsten gleiche Daten wie im Training!

        self.forward_propagation(e=0)

        h_one_hot = (softmax(self.o_all))  # predicted_output
        f1_scores, f1_scores_mean = calculate_f1_scores(h_one_hot, self.y_validation)  # Übergeben berechnete softmax Werte und ground truth

        return f1_scores, f1_scores_mean

    def confusion(self, data_x, ground_truth):
        self.x_validation = data_x
        self.y_validation = ground_truth
        f1_scores, f1_scores_mean = self.validation()
        h_one_hot = np.zeros((len(self.o_all),))
        for i in range(len(self.o_all)):
            h_one_hot[i] = np.argmax(self.o_all[i])
        gtruth = np.asarray(np.where(ground_truth == 1))
        counter_array = []
        for i in range(len(h_one_hot)):
            counter_array.append(str(h_one_hot[i]) + str(gtruth[1, i]))

        gesture_num = len(self.o_all[0])
        confusion_matrix = np.asarray(3*[(gesture_num * gesture_num)*[0]])
        for i in range(gesture_num):
            confusion_matrix[0][(i * gesture_num):((i + 1) * gesture_num)] = i
        for i in range(gesture_num):
            confusion_matrix[1][gesture_num * np.arange(gesture_num) + i] = i
        for i in range(len(confusion_matrix[2])):
            confusion_matrix[2, i] = counter_array.count(str(confusion_matrix[0, i]) + '.0' + str(confusion_matrix[1,
                                                                                                                   i]))

        g = sns.relplot(
            x=confusion_matrix[0], y=confusion_matrix[1],
            sizes=(0, 1000),
            size=confusion_matrix[2], legend=False
        )
        g.set(xscale="linear", yscale="linear")
        g.ax.xaxis.grid(True, "minor", linewidth=.5)
        g.ax.yaxis.grid(True, "minor", linewidth=.5)
        g.axes[0][0].hlines(y=[np.arange((gesture_num-1))+0.5], ls='solid', xmin=-0.5, xmax=(gesture_num-0.5),
                            color='grey')
        g.axes[0][0].vlines(x=[np.arange((gesture_num-1))+0.5], ls='solid', ymin=-0.5, ymax=(gesture_num-0.5),
                            color='grey')
        g.set_axis_labels("PREDICTED OUTPUT", "ACTUAL OUTPUT")
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks (axis notation) at regular intervals
        g.ax.xaxis.set_major_locator(loc)
        g.ax.yaxis.set_major_locator(loc)

        plt.show()
        return f1_scores, f1_scores_mean, g
