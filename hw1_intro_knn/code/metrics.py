import numpy as np


def check_0_div(func):
    def wrapper_check_0_div(*args, **kwargs):
        value = func(*args, **kwargs)
        if np.isinf(value):
            return 1
        return value
    return wrapper_check_0_div


@check_0_div
def precision(tp, fp):
    return tp / (tp + fp)


@check_0_div
def recall(tp, fn):
    return tp / (tp + fn)


@check_0_div
def accuracy(tp, tn, n):
    return (tp + tn) / n


@check_0_div
def f1(tp, fp, fn):
    return 2*tp/(2*tp + fp + fn)


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision_val, recall_val, accuracy_val, f1_val - classification metrics
    """
    n = y_pred.shape[0]
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    fp = np.sum((y_pred != y_true) & (y_true == 0))
    tn = np.sum((y_pred == y_true) & (y_true == 0))
    fn = np.sum((y_pred != y_true) & (y_true == 1))
    precision_val = precision(tp, fp)
    recall_val = recall(tp, fn)
    accuracy_val = accuracy(tp, tn, n)
    f1_val = f1(tp, fp, fn)
    # print(f"Precision: {precision_val}")
    # print(f"Recall: {recall_val}")
    # print(f"Accuracy: {accuracy_val}")
    # print(f"F1: {f1_val}")
    return precision_val, recall_val, accuracy_val, f1_val


@check_0_div
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    n = y_pred.shape[0]
    return np.sum((y_pred == y_true))/n


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - np.sum(np.square(y_true-y_pred))/np.sum(np.square(y_true-np.mean(y_true)))
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    n = len(y_pred)
    mse = np.sum(np.square(y_true-y_pred))/n
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolute error
    """
    n = len(y_pred)
    mae = np.sum(np.abs(y_true-y_pred))/n
    return mae
