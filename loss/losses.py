import torch
import torch.nn as nn

from config import FOCAL_EPSILON

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, ignore_index=None, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = FOCAL_EPSILON
        self.smooth = FOCAL_EPSILON

    def forward(self, output, target):
        pos_mask = (target >= self.eps).float()
        neg_mask = (target < self.eps).float()

        pos_output = output * pos_mask
        neg_output = output * neg_mask
        pos_target = target * pos_mask
        neg_target = target * neg_mask

        pos_loss = torch.pow(1 - pos_output, self.alpha) * torch.log(pos_output + self.smooth)
        pos_loss = pos_loss * pos_target

        neg_loss = torch.pow(1 - neg_target, self.beta) * torch.pow(neg_output, self.alpha) * torch.log(1 - neg_output + self.smooth)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        loss = -1 * loss

        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice