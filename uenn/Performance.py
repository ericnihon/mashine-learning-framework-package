import numpy as np


def f1_score(h, y):
    h = np.round(h)

    true_positives = (h == 1) & (y == 1)
    false_positives = (h == 1) & (y == 0)
    false_negatives = (h == 0) & (y == 1)

    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_f1_scores(predictions, ground_truth):
    f1_scores = []

    one_hot = np.argmax(predictions, axis=1)
    predictions = np.eye(len(ground_truth[0] - 1))[one_hot]

    for output_label in range(len(ground_truth[0])):
        f1 = f1_score(predictions[:, output_label], ground_truth[:, output_label])
        f1_scores.append(f1)

    f1_score_mean = np.mean(f1_scores)
    return f1_scores, f1_score_mean
