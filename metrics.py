import torch
import numpy as np


# --- True positives --------------------------------------------------------------------------
def true_positives(y_pred, y_truth, ith_class):
    true_labels_indices = np.array([i for i, y in enumerate(y_truth) if y.item() == ith_class])
    tp = torch.sum(y_pred[true_labels_indices] == y_truth[true_labels_indices]).item()
    
    return tp

# --- True negatives --------------------------------------------------------------------------
def true_negatives(y_pred, y_truth, ith_class):
    
    true_labels_indices = np.array([i for i, y in enumerate(y_truth) if y.item() != ith_class])
    tn = torch.sum(y_pred[true_labels_indices] == y_truth[true_labels_indices]).item()
    
    return tn

# --- False positives --------------------------------------------------------------------------
def false_positives(y_pred, y_truth, ith_class):
    
    true_labels_indices = np.array([i for i, y in enumerate(y_truth) if y.item() != ith_class])
    fp = torch.sum(y_pred[true_labels_indices] == ith_class).item()
    return fp

# --- False negatives --------------------------------------------------------------------------
def false_negatives(y_pred, y_truth, ith_class):
    
    true_labels_indices = np.array([i for i, y in enumerate(y_truth) if y.item() == ith_class])
    fn = torch.sum(y_pred[true_labels_indices] == ith_class).item()
    
    return fn

# --- Class accuracy ---------------------------------------------------------------------------

def accuracy(y_pred, y_truth, ith_class):
    tp, tn = true_positives(y_pred, y_truth, ith_class), true_negatives(y_pred, y_truth, ith_class)
    fp, fn = false_positives(y_pred, y_truth, ith_class), false_negatives(y_pred, y_truth, ith_class)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


# --- Recall ---------------------------------------------------------------------------

def recall(y_pred, y_truth, ith_class):
    tp, tn = true_positives(y_pred, y_truth, ith_class), true_negatives(y_pred, y_truth, ith_class)
    fp, fn = false_positives(y_pred, y_truth, ith_class), false_negatives(y_pred, y_truth, ith_class)

    recall_ = tp / (tp + fn)

    return recall_

# --- Percision ---------------------------------------------------------------------------

def precision(y_pred, y_truth, ith_class):
    tp, tn = true_positives(y_pred, y_truth, ith_class), true_negatives(y_pred, y_truth, ith_class)
    fp, fn = false_positives(y_pred, y_truth, ith_class), false_negatives(y_pred, y_truth, ith_class)

    precision_ = tp / (tp + fp)

    return precision_


# --- Class accuracy ---------------------------------------------------------------------------

def f_score(y_pred, y_truth, ith_class):
    precision_ = precision(y_pred, y_truth, ith_class)
    recall_ = recall(y_pred, y_truth, ith_class)
    
    f_score_ = 2 * precision_ * recall_ / (precision_ + recall_)

    return f_score_