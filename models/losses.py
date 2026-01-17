"""
Additional loss functions for improving training on imbalanced data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance problem.
    
    Focal Loss focuses on hard examples, reducing the weight of easy examples.
    Useful when some classes (e.g., N1) occur less frequently than others.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        """
        Args:
            alpha: Weighting factor for class balancing (can use class weights)
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            weight: Class weights (optional)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - model logits
            targets: (N,) - class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for regularization and improved generalization.
    
    Instead of hard labels [0, 0, 1, 0, 0] uses soft labels [0.05, 0.05, 0.8, 0.05, 0.05]
    """
    
    def __init__(self, num_classes=5, smoothing=0.1, weight=None):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (usually 0.1)
            weight: Class weights (optional)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - model logits
            targets: (N,) - class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
            if self.weight is not None:
                # Apply class weights
                class_weights = self.weight[targets].unsqueeze(1)
                true_dist = true_dist * class_weights
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
