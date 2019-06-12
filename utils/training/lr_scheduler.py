from torch.optim.lr_scheduler import _LRScheduler


class LinearLR(_LRScheduler):

    def __init__(self, optimizer, initial_lr=-1, nr_epochs=-1, initial_hold=0, last_epoch=-1):

        assert nr_epochs >= initial_hold
        assert initial_lr > 0
        assert nr_epochs >= 0

        self.initial_hold = initial_hold
        self.step_size = (-initial_lr) / (nr_epochs - initial_hold)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.initial_hold:
            return self.base_lrs
        return [group['lr'] + self.step_size for group in self.optimizer.param_groups]
