import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy and F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        "accuracy": accuracy,
        "f1_score": f1
    }

def get_confusion_matrix(y_true, y_pred):
    """
    Generate a confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
