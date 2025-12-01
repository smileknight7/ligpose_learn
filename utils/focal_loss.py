import torch
import torch.nn.functional as F
#损失函数脚本

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, bi=False):
        if bi:
            ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        else:
            ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss
