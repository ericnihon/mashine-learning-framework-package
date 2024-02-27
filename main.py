from uenn.DataTools import importing, scaling, preparing
from uenn.NeuralNet import NN
from uenn.Plotter import error_plots, f1_plots
from uenn.PCA import pca
from uenn.Saver import save_thetas, save_plots
import os


# Daten Import, Skalierung, Aufbereitung, PCA
print('loading data\n')
X, y = importing(path=os.getcwd() + '\\InputData', mnist=True)  # returns np.array || für MNIST pd.DataFrame

print('scaling data\n')
X = scaling(X)

print('preparing data\n')
X, y = preparing(X, y)  # returns dictionaries; benötigt np.array oder pd.DataFrame

print('calculating PCA\n')
X = pca(X)

# Neural Net erstellen
print('create neural network\n')
nn = NN(input_size=len(X['train'][0]), layer_number=2, layer_size=23, output_size=len(y['train'][0]))

# Training starten
print('start training\n')
tuned_thetas, errors_mean, errors_output, f1_scores_list, f1_scores_list_mean = nn.train(X, y, alpha=0.05, epochs=20,
                                                                                         batch=2000)

# Confusion Matrix
f1_output, f1_mean, fig_confusion_matrix = nn.confusion(X['validation'], y['validation'])

# Thetas abspeichern
save_thetas(tuned_thetas)

# Letzte F Werte anzeigen
print(f1_scores_list[-1])
print(f1_scores_list_mean[-1])

# Plots erstellen
fig_mean_error, fig_output_error = error_plots(errors_mean, errors_output)
fig_mean_f1, fig_output_f1, fig_bar_f1 = f1_plots(f1_scores_list_mean, f1_scores_list)

# Plot abspeichern
save_plots(fig_confusion_matrix)
