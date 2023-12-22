import torch
import torch.nn as nn

class DRLoss(nn.Module):
    def __init__(self):
        super(DRLoss, self).__init__()

        self.DR_COMPARISON_TARGET = [
            1, 2, 3, 4, 5, 6, 7, 1234,
            0, 8, 9, 10, 11, 12, 13, 14
        ]
        
    def forward(self, pred_center):
        pred_center = pred_center.squeeze(0)
        dists = []

        for idx in range(pred_center.shape[0]):
            if idx == 7:
                continue

            near_idx = self.DR_COMPARISON_TARGET[idx]
            cur_center = pred_center[idx]
            near_center = pred_center[near_idx]

            x = torch.dist(
                cur_center, near_center, 2
            )

            dists.append(x)

        ddists = []
        for didx in range(len(dists) - 1):
            near_didx = didx + 1

            ddist = (dists[near_didx] - dists[didx])
            ddists.append(ddist)

        dr_losses = []
        for ddidx in range(len(ddists) - 1):
            near_ddidx = ddidx + 1

            dddist = torch.dist(
                ddists[near_ddidx], ddists[ddidx], 2
            )
            dr_losses.append(dddist)

        dr_losses = torch.stack(dr_losses, dim=0)
        dr_losses = dr_losses.mean()
        dr_losses = 0.0001 * dr_losses

        return dr_losses