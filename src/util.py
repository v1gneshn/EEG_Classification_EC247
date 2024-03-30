import numpy as np
import torch


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of classification.
    
    Parameters:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        
    Returns:
        float: Accuracy of classification.
    """
    # Ensure both tensors are on the same device
    device = y_true.device
    y_pred = y_pred.to(device)
    
    # Convert predicted probabilities to predicted labels
    y_pred_labels = torch.argmax(y_pred, dim=1)
    
    # Compare predicted labels with true labels and calculate accuracy
    correct = (y_pred_labels == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    
    return accuracy