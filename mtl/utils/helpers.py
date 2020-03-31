from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus


def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return PolyLR(optimizer, cfg.lr_scheduler_power, cfg.num_epochs)
    else:
        raise NotImplementedError


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, power, num_steps, last_epoch=-1):
        self.power = power
        self.num_steps = num_steps
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - min(self.last_epoch, self.num_steps-1) / self.num_steps) ** self.power
                for base_lr in self.base_lrs]
