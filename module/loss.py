import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class GetCriterion:
    def __init__(self, loss, start_weight=None, target=None, n_epoch=None):
        """
        loss          : CrossEntropyなどの種類
        start_weight  : class weight初期値
        target        : class weightを変化させる場合の最終目標
        n_epoch       : 最終epoch数
        """
        self.loss = loss
        self.start_weight = start_weight
        self.target = target
        self.n_epoch = n_epoch
        self.step = 0

        if start_weight is not None and target is not None:
            self.step_weight = (target - start_weight)
            self.step_weight /= n_epoch    
        else:
            self.step_weight = None

    def get_weight(self):
        if self.step < self.n_epoch:
            weight = self.start_weight + self.step * self.step_weight
        else:
            weight = self.target
        self.step += 1
        return weight

    def __call__(self, device):
        if self.target is not None:
            weight = self.get_weight()
            weight = weight.to(device)
        else:
            weight = self.start_weight

        if self.loss == "CE":
            criterion = nn.CrossEntropyLoss(weight=weight)
        elif self.loss == "focal":
            criterion = FocalLoss(alpha=weight)
        return criterion