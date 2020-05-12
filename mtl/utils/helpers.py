from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus
from mtl.models.model_branched import ModelBranched
from mtl.models.model_distillation import ModelDistillation

def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
        'branched': ModelBranched,
        'distillation': ModelDistillation
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
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / (cfg.num_epochs-1)) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError
