import numpy as np
import torch
import torch.nn as nn

# TODO 名前このまま？
class CustomLoss(nn.Module):
    def __init__(self, swing_ratio=0.7):
        super().__init__()
        self.swing_ratio = swing_ratio

    def forward(self, outputs, targets):
        loss = 0
        for pred, target in zip(outputs, targets):
            if target != 0:
                loss += - self.swing_ratio * torch.log(pred[target])
            else:
                loss += - (1 - self.swing_ratio) * torch.log(pred[target])
        loss = loss / len(targets)
        return loss


            # if (pred.argmax() == 1) and (target == 0):
            #     loss += - 2 * self.swing_ratio * torch.log(pred[target])