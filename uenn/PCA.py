import numpy as np


def pca(x: object, cutoff=0.99, path=False):
    # cutoff muss/sollte zwischen 0 und 1 liegen. Größer entspricht alle Hauptkomponenten werden genommen negativ
    # entspricht keine Hauptkomponenten werden genommen
    if cutoff < 0 or cutoff >= 1:
        print('cutoff should be set between 0 and 1')
        return x
    try:
        x_t = x['train'].T  #im Grunde kennen wir keinen Validation Datensatz, daher schauen wir PCA nur train an.
    except:
        print('x is not a dictionary with \'train\' key, please provide a dictionary with \'train\' and \'validation\' '
              'keys. You may use the preparing() method')
    cov_features_matrix = np.cov(x_t)
    eigenvalues, eigenvectors = np.linalg.eig(cov_features_matrix)

    # Prozente berechnen
    explained_variance = []
    for i in eigenvalues:
        explained_variance.append(i/sum(eigenvalues))

    i = 0
    criterium = 0
    while criterium < cutoff:
        criterium += explained_variance[i]
        i += 1

    eigenvector_matrix = eigenvectors[:i]
    x = feature_converter(x, eigenvector_matrix, eigenvectors)

    if path:
        save_eigenvector_matrix(eigenvector_matrix, path)

    return x


def feature_converter(x: object, eigenvector_matrix: object, eigenvectors: object):
    transformed_eigmatrix = eigenvector_matrix.T
    x['train'] = x['train'] @ transformed_eigmatrix
    x['validation'] = x['validation'] @ transformed_eigmatrix
    return x


def save_eigenvector_matrix(eigenvector_matrix: object, path: str):
    np.save(path + '\\eigenvector_matrix', eigenvector_matrix)
