import numpy as np


def softmax(v):
    e = np.e**v
    e_sums = e.sum(axis=1)[:, np.newaxis]
    result = e / e_sums
    return result


def categorical_cross_entropy(h_one_hoty, y_one_hot):
    # we need to clip because `h` must never be <= 0, which
    # doesn't happen in theory, but will happen in praxis because of
    # automatic rounding
    h_one_hot = softmax(h_one_hoty)
    h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)
    errors_mean = np.sum(-(np.sum(y_one_hot * np.log(h_one_hot), axis=1))) / len(h_one_hot)

    return errors_mean


def categorical_cross_entropy_per_output(h_one_hot: np.array, y_one_hot: np.array):
    # we need to clip because `h` must never be <= 0, which
    # doesn't happen in theory, but will happen in praxis because of
    # automatic rounding
    h_one_hot = softmax(h_one_hot)
    h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)

    errors_per_neuron = []
    for neuron in range(len(y_one_hot[0])):
        # Nimmt für einzelnes Output Neuronen softmax Wert des berechneten Output Werts
        h_one_hot_current_neuron = h_one_hot[:, neuron]
        # Nimmt für einzelnes Output Neuron ground truth Wert
        y_one_hot_current_neuron = y_one_hot[:, neuron]

        current_neuron_error = -(np.sum(y_one_hot_current_neuron * np.log(h_one_hot_current_neuron))) / len(h_one_hot)

        errors_per_neuron.append(current_neuron_error)

    errors_per_neuron = np.asarray(errors_per_neuron)
    return errors_per_neuron


def mean_squared_error(h_one_hot, y_one_hot):
    result = np.mean((h_one_hot - y_one_hot)**2)
    return result
