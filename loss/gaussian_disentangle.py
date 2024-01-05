import torch
import torch.nn as nn


class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()

        self.COMPARISON_TARGET = [8, 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, -1,
             24, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, -1]
        
    def forward(self, heatmaps):
        '''
        heatmap (shape : [b, 16, d, h, w])
        '''
        losses = []
        heatmaps = heatmaps[0]
        for i, heat in enumerate(heatmaps):
            comp_targ = self.COMPARISON_TARGET[i]
            if comp_targ == -1:
                continue
            else:
                losses.append(heat * heatmaps[comp_targ])

        losses = torch.stack(losses, dim=0)
        losses = losses.mean()
        return losses