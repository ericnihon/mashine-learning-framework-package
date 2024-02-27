import numpy as np
import pandas as pd
import os
import fnmatch
from sklearn.datasets import fetch_openml
from uenn import Scalers


def importing(path: str, ground_truth='ground_truth', mnist=False):
    if mnist:
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype(int)
        y_one_hot = np.eye(y.max() + 1)[y]
        return x, y_one_hot

    list_all_csv = []
    for path, directory_list, file_list in os.walk(path):
        for name in fnmatch.filter(file_list, '*.csv'):
            list_all_csv.append(os.path.join(path, name))

    array_x = []
    array_y = []
    for file in list_all_csv:
        current_df = pd.read_csv(file, header=[0])
        current_y = current_df[ground_truth]
        current_x = current_df.drop(ground_truth, axis=1)     # entfernen ground_truth für X
        array_x.append(current_x.to_numpy())   # enthält alle X
        array_y.append(current_y.to_numpy())   # enthält alle ground_truths

    array_x = np.asarray(array_x)
    array_y = np.asarray(array_y)

    # Verketten der Arrays, dass weniger Dimensionen vorhanden sind
    arr_x = array_x[0]
    for x in range(len(array_x) - 1):
        arr_x = np.concatenate((arr_x, array_x[x + 1]), axis=0)

    arr_y = array_y[0]
    for y in range(len(array_y) - 1):
        arr_y = np.concatenate((arr_y, array_y[y + 1]), axis=0)
    arr_y = arr_y.astype(int)
    arr_y = np.eye(arr_y.max() + 1)[arr_y]

    return arr_x, arr_y


def scaling(x: object, scaler='standardized'):
    if scaler == 'standardized':
        scaler = Scalers.StandardScaler()
    elif scaler == 'normalized':
        scaler = Scalers.NormalScaler()
    else:
        print('scaler needs to be defined as \'standardized\' or \'normalized\'')
    scaler.fit(x)
    scaled_data = scaler.transform(x)
    return scaled_data


def preparing(x: object, y: object, ratio=0.7):
    sample_size = len(x)
    train_size = int(sample_size * ratio)

    # Indices Arrays erstellen, um train und validation data auf gleiche Art randomisieren zu können
    # (indices array shuffeln nicht x direkt, sonst ist y nicht zuordnungsbar)
    indices = np.arange(sample_size)
    np.random.shuffle(indices)

    # train/validation Indices
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:]

    x = np.asarray(x)
    # Sets erstellen
    x = {
        'train': x[train_indices],
        'validation': x[validation_indices]
    }
    y = {
        'train': y[train_indices],
        'validation': y[validation_indices]
    }
    return x, y
