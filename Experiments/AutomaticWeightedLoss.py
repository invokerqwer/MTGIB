# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2,use_uncertainty=True,PRIMARY_TASK_INDEX=0, primary_task_weight=1.2):
        super(AutomaticWeightedLoss, self).__init__()
        self.use_uncertainty = use_uncertainty
        print("use multi uncertainty loss ", self.use_uncertainty)
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.PRIMARY_TASK_INDEX = PRIMARY_TASK_INDEX
        self.primary_task_weight = primary_task_weight

    def forward(self, *x):
        loss_sum = 0
        if self.use_uncertainty:
            for i, loss in enumerate(x):
                if i == self.PRIMARY_TASK_INDEX:
                    # loss_sum += self.primary_task_weight * (0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
                    loss_sum += (self.primary_task_weight * 0.5 / (self.params[i] ** 2) )* loss + torch.log(1 + self.params[i] ** 2)
                else:
                    loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        else:
            loss_sum = sum(x)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(f"uncertainty weight params values: {[param.data.cpu().numpy() for param in awl.parameters()]}")
    