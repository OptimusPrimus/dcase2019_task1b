import torch.nn as nn

class CrossEntropyLoss():

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, pred, target):

        if len(target.shape) == 2:
            return - (pred.log_softmax(dim=-1) * target).sum(dim=-1).mean()
        else:
            return self.criterion(pred, target)
