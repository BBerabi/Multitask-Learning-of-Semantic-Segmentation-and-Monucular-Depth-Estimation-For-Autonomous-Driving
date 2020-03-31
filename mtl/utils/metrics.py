import torch


def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes, ignore_label):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y_lbl), 'Inputs must be torch tensors'
    assert y_lbl.device == y_hat_lbl.device, 'Input tensors have different device placement'

    assert y_hat_lbl.dim() == 3 or y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1
    assert y_lbl.dim() == 3 or y_lbl.dim() == 4 and y_lbl.shape[1] == 1
    if y_hat_lbl.dim() == 4:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y_lbl.dim() == 4:
        y_lbl = y_lbl.squeeze(1)

    mask = y_lbl != ignore_label
    y_hat_lbl = y_hat_lbl[mask]
    y_lbl = y_lbl[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + num_classes * y_lbl
    bincount_2d = torch.bincount(x.long(), minlength=num_classes ** 2)
    assert bincount_2d.numel() == num_classes ** 2, 'Internal error'
    conf = bincount_2d.view((num_classes, num_classes)).long()
    return conf


def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = 100 * diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class


class MetricsSemseg:
    def __init__(self, num_classes, ignore_label, class_names):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names
        self.metrics_acc = None

    def reset(self):
        self.metrics_acc = None

    def update_batch(self, y_hat_lbl, y_lbl):
        with torch.no_grad():
            metrics_batch = semseg_compute_confusion(y_hat_lbl, y_lbl, self.num_classes, self.ignore_label).cpu()
            if self.metrics_acc is None:
                self.metrics_acc = metrics_batch
            else:
                self.metrics_acc += metrics_batch

    def get_metrics_summary(self):
        iou_mean, iou_per_class = semseg_accum_confusion_to_iou(self.metrics_acc)
        out = {self.class_names[i]: iou for i, iou in enumerate(iou_per_class)}
        out['mean_iou'] = iou_mean
        return out


def depth_metrics_calc_one(y_hat_meters, y_meters):
    assert y_hat_meters.dim() == 2 and y_hat_meters.shape == y_meters.shape

    metrics = {}

    valid = y_meters == y_meters
    y_hat_meters = y_hat_meters[valid].double()
    y_meters = y_meters[valid].double()
    n = y_meters.numel()
    if n == 0:
        return None, False

    y_meters = y_meters.clamp(min=0.01)
    y_hat_meters = y_hat_meters.clamp(min=0.01)
    y_logmeters = y_meters.log()
    y_hat_logmeters = y_hat_meters.log()

    # log meters
    d_diff_log = y_logmeters - y_hat_logmeters
    d_diff_log_sum = d_diff_log.sum()

    metrics['log_mae'] = d_diff_log.abs().mean()
    normalized_squared_log = (d_diff_log * d_diff_log).mean()

    d_err = (y_meters - y_hat_meters).abs()
    d_err_squared = d_err * d_err

    metrics['mae'] = d_err.mean()
    metrics['rmse'] = d_err_squared.mean().sqrt()
    metrics['rel'] = (d_err / y_meters).mean()
    metrics['rel_squared'] = (d_err_squared / (y_meters * y_meters)).mean()

    # delta thresholds
    y_div_y_hat = y_meters / y_hat_meters
    mask_delta_1 = (y_div_y_hat > (1 / 1.25 ** 1)) & (y_div_y_hat < (1.25 ** 1))
    mask_delta_2 = (y_div_y_hat > (1 / 1.25 ** 2)) & (y_div_y_hat < (1.25 ** 2))
    mask_delta_3 = (y_div_y_hat > (1 / 1.25 ** 3)) & (y_div_y_hat < (1.25 ** 3))
    metrics['delta1'] = 100 * mask_delta_1.double().mean()
    metrics['delta2'] = 100 * mask_delta_2.double().mean()
    metrics['delta3'] = 100 * mask_delta_3.double().mean()

    d_err_inv = ((1 / y_meters) - (1 / y_hat_meters)).abs()
    metrics['inv_mae'] = d_err_inv.mean()
    metrics['inv_rmse'] = (d_err_inv ** 2).mean().sqrt()

    metrics['log_rmse'] = normalized_squared_log.sqrt()
    metrics['si_log_rmse'] = 100 * (normalized_squared_log - d_diff_log_sum * d_diff_log_sum / (n * n)).sqrt()

    return metrics, True


def depth_metrics_calc_batch(y_hat_meters, y_meters):
    assert torch.is_tensor(y_hat_meters) and torch.is_tensor(y_meters), 'Inputs must be torch tensors'
    assert y_meters.device == y_hat_meters.device, 'Input tensors must have same device placement'

    assert y_hat_meters.dim() == 3 or y_hat_meters.dim() == 4 and y_hat_meters.shape[1] == 1
    assert y_meters.dim() == 3 or y_meters.dim() == 4 and y_meters.shape[1] == 1
    if y_hat_meters.dim() == 4:
        y_hat_meters = y_hat_meters.squeeze(1)
    if y_meters.dim() == 4:
        y_meters = y_meters.squeeze(1)

    out, cnt = None, 0
    for i in range(y_hat_meters.shape[0]):
        tmp, valid = depth_metrics_calc_one(y_hat_meters[i, :, :], y_meters[i, :, :])
        if valid:
            if out is None:
                out = tmp
            else:
                out = {k: v + tmp[k] for k, v in out.items()}
            cnt += 1
    return out, cnt


class MetricsDepth:
    def __init__(self):
        self.metrics_acc = None
        self.counter = 0

    def reset(self):
        self.metrics_acc = None
        self.counter = 0

    def update_batch(self, y_hat_meters, y_meters):
        with torch.no_grad():
            metrics_batch, cnt = depth_metrics_calc_batch(y_hat_meters, y_meters)
            metrics_batch = {k: v.cpu() for k, v in metrics_batch.items()}
            if self.metrics_acc is None:
                self.metrics_acc = metrics_batch
            else:
                self.metrics_acc = {k: v + self.metrics_acc[k] for k, v in metrics_batch.items()}
        self.counter += cnt

    def get_metrics_summary(self):
        return {k: v / self.counter for k, v in self.metrics_acc.items()}
