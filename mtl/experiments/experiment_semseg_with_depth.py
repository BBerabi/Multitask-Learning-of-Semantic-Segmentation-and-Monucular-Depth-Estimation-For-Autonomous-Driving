import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from mtl.datasets.definitions import *
from mtl.losses.loss_regression import LossRegression
from mtl.utils.metrics import MetricsSemseg, MetricsDepth
from mtl.utils.helpers import resolve_optimizer, resolve_dataset_class, resolve_model_class, resolve_lr_scheduler
from mtl.utils.transforms import get_transforms
from mtl.utils.visualization import compose


class ExperimentSemsegDepth(pl.LightningModule):
    def __init__(self, cfg):
        super(ExperimentSemsegDepth, self).__init__()
        self.cfg = cfg

        dataset_class = resolve_dataset_class(cfg.dataset)
        self.datasets = {
            split: dataset_class(cfg.dataset_root, split, integrity_check=False)
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')

        self.semseg_num_classes = self.datasets[SPLIT_TRAIN].semseg_num_classes
        self.semseg_ignore_label = self.datasets[SPLIT_TRAIN].semseg_ignore_label
        self.semseg_class_names = self.datasets[SPLIT_TRAIN].semseg_class_names
        self.semseg_class_colors = self.datasets[SPLIT_TRAIN].semseg_class_colors
        self.rgb_mean = self.datasets[SPLIT_TRAIN].rgb_mean
        self.rgb_stddev = self.datasets[SPLIT_TRAIN].rgb_stddev
        self.depth_meters_mean = self.datasets[SPLIT_TRAIN].depth_meters_mean
        self.depth_meters_stddev = self.datasets[SPLIT_TRAIN].depth_meters_stddev
        self.depth_meters_min = self.datasets[SPLIT_TRAIN].depth_meters_min
        self.depth_meters_max = self.datasets[SPLIT_TRAIN].depth_meters_max

        outputs_descriptor = {
            MOD_SEMSEG: self.semseg_num_classes,
            MOD_DEPTH: 1,
        }

        model_class = resolve_model_class(cfg.model_name)
        self.net = model_class(cfg, outputs_descriptor)
        print(self.net)

        self.datasets[SPLIT_TRAIN].set_transforms(get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            geom_scale_min=cfg.aug_geom_scale_min,
            geom_scale_max=cfg.aug_geom_scale_max,
            geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
            geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
            geom_reflect=cfg.aug_geom_reflect,
            crop_random=cfg.aug_input_crop_size,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
        ))
        self.transforms_val_test = get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            crop_for_passable=32,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
        )
        self.datasets[SPLIT_VALID].set_transforms(self.transforms_val_test)
        self.datasets[SPLIT_TEST].set_transforms(self.transforms_val_test)

        self.loss_semseg = torch.nn.CrossEntropyLoss(ignore_index=self.semseg_ignore_label)
        self.loss_depth = LossRegression()

        self.metrics_semseg = MetricsSemseg(self.semseg_num_classes, self.semseg_ignore_label, self.semseg_class_names)
        self.metrics_depth = MetricsDepth()

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            rgb = rgb.cuda()
            y_semseg_lbl = y_semseg_lbl.cuda()
            y_depth = y_depth.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]
        y_hat_depth = y_hat[MOD_DEPTH]

        if type(y_hat_semseg) is list:
            # deep supervision scenario: penalize all predicitons in the list and average losses
            loss_semseg = sum([self.loss_semseg(y_hat_semseg_i, y_semseg_lbl) for y_hat_semseg_i in y_hat_semseg])
            loss_depth = sum([self.loss_depth(y_hat_depth_i, y_depth) for y_hat_depth_i in y_hat_depth])
            loss_semseg = loss_semseg / len(y_hat_semseg)
            loss_depth = loss_depth / len(y_hat_depth)
            y_hat_semseg = y_hat_semseg[-1]
            y_hat_depth = y_hat_depth[-1]
        else:
            loss_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
            loss_depth = self.loss_depth(y_hat_depth, y_depth)

        loss_total = self.cfg.loss_weight_semseg * loss_semseg + self.cfg.loss_weight_depth * loss_depth

        tensorboard_logs = {
            'loss_train/semseg': loss_semseg,
            'loss_train/depth': loss_depth,
            'loss_train/total': loss_total,
        }

        if self.can_visualize():
            self.visualize(batch, y_hat_semseg, y_hat_depth, batch[MOD_ID], 'train/batch_crops')

        return {
            'progress_bar': {
                'semseg': loss_semseg,
                'depth': loss_depth,
            },
            'loss': loss_total,
            'log': tensorboard_logs
        }

    def inference_step(self, batch):
        rgb = batch[MOD_RGB]

        if torch.cuda.is_available():
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]
        y_hat_depth = y_hat[MOD_DEPTH]

        if type(y_hat_semseg) is list:
            y_hat_semseg = y_hat_semseg[-1]
            y_hat_depth = y_hat_depth[-1]

        y_hat_semseg_lbl = y_hat_semseg.argmax(dim=1)
        y_hat_depth_meters = (
                y_hat_depth * self.depth_meters_stddev + self.depth_meters_mean
        ).clamp(self.depth_meters_min, self.depth_meters_max)

        return y_hat_semseg, y_hat_semseg_lbl, y_hat_depth, y_hat_depth_meters

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl, y_hat_depth, y_hat_depth_meters = self.inference_step(batch)

        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            y_semseg_lbl = y_semseg_lbl.cuda()
            y_depth = y_depth.cuda()

        y_depth_meters = y_depth * self.depth_meters_stddev + self.depth_meters_mean

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
        loss_val_depth = self.loss_depth(y_hat_depth, y_depth)
        loss_val_total = loss_val_semseg + loss_val_depth

        self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)
        self.metrics_depth.update_batch(y_hat_depth_meters, y_depth_meters)

        return {
            'semseg': loss_val_semseg.cpu().item(),
            'depth': loss_val_depth.cpu().item(),
            'total': loss_val_total.cpu().item(),
        }

    def validation_end(self, outputs):
        self.observer_step()

        loss_val_semseg = sum([a['semseg'] for a in outputs]) / len(outputs)
        loss_val_depth = sum([a['depth'] for a in outputs]) / len(outputs)
        loss_val_total = sum([a['total'] for a in outputs]) / len(outputs)

        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metrics_depth = self.metrics_depth.get_metrics_summary()
        self.metrics_depth.reset()

        metric_semseg = metrics_semseg['mean_iou']
        metric_depth = metrics_depth['si_log_rmse']

        metric_total = metric_semseg - metric_depth
        metric_grader = (metric_semseg - 50).clamp(min=0) + (50 - metric_depth).clamp(min=0)

        tensorboard_logs = {
            'metrics_summary/semseg': metric_semseg,
            'metrics_summary/depth': metric_depth,
            'metrics_summary/total': metric_total,
            'metrics_summary/grader': metric_grader,
            'loss_val/semseg': loss_val_semseg,
            'loss_val/depth': loss_val_depth,
            'loss_val/total': loss_val_total,
            'LR': self.trainer.lr_schedulers[0].get_last_lr()[0],
        }
        tensorboard_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})
        tensorboard_logs.update({f'metrics_task_depth/{k}': v for k, v in metrics_depth.items()})

        return {
            'progress_bar': {
                'semseg': metric_semseg,
                'depth': metric_depth,
            },
            'log': tensorboard_logs,
            'metric': metric_total
        }

    def test_step(self, batch, batch_nb):
        _, y_hat_semseg_lbl, _, y_hat_depth_meters = self.inference_step(batch)
        path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        path_pred_semseg = os.path.join(path_pred, MOD_SEMSEG)
        path_pred_depth = os.path.join(path_pred, MOD_DEPTH)
        if batch_nb == 0:
            os.makedirs(path_pred_semseg)
            os.makedirs(path_pred_depth)
        split_test = SPLIT_TEST
        for i in range(y_hat_semseg_lbl.shape[0]):
            sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
            path_file_semseg = os.path.join(path_pred_semseg, f'{sample_name}.png')
            path_file_depth = os.path.join(path_pred_depth, f'{sample_name}.png')
            pred_semseg = y_hat_semseg_lbl[i]
            pred_depth = y_hat_depth_meters[i]
            self.datasets[split_test].save_semseg(
                path_file_semseg, pred_semseg, self.semseg_class_colors, self.semseg_ignore_label
            )
            self.datasets[split_test].save_depth(
                path_file_depth, pred_depth, out_of_range_policy='clamp_to_range'
            )

    def test_end(self, outputs):
        return {}

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [lr_scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

    def create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=True,
            drop_last=False,
        )

    @pl.data_loader
    def val_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_VALID)

    @pl.data_loader
    def test_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_TEST)

    def visualize(self, batch, y_hat_semseg, y_hat_depth, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_semseg_lbl = y_hat_semseg.cpu().detach().argmax(dim=1)
        y_hat_depth = y_hat_depth.cpu().detach()
        visualization_plan = [
            (MOD_RGB, batch[MOD_RGB], rgb_tags),
            (MOD_SEMSEG, batch[MOD_SEMSEG], 'GT SemSeg'),
            (MOD_SEMSEG, y_hat_semseg_lbl, 'Prediction SemSeg'),
            (MOD_DEPTH, batch[MOD_DEPTH], 'GT Depth'),
            (MOD_DEPTH, y_hat_depth, 'Prediction Depth'),
        ]
        vis = compose(
            visualization_plan,
            self.cfg,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            semseg_color_map=self.semseg_class_colors,
            semseg_ignore_label=self.semseg_ignore_label,
        )
        self.logger.experiment.add_image(tag, vis, self.global_step)
        # visualize depth histograms to see how far the distribution of values is away from gaussian
        depth_normalized = batch[MOD_DEPTH]
        depth_normalized = depth_normalized[depth_normalized == depth_normalized]
        depth_meters = depth_normalized * self.depth_meters_stddev + self.depth_meters_mean
        y_hat_depth_meters = y_hat_depth * self.depth_meters_stddev + self.depth_meters_mean
        self.logger.experiment.add_histogram('GT_Depth_Normalized', depth_normalized, self.trainer.global_step, bins=64)
        self.logger.experiment.add_histogram('GT_Depth_Meters', depth_meters, self.trainer.global_step, bins=64)
        self.logger.experiment.add_histogram('Pred_Depth_Normalized', y_hat_depth, self.trainer.global_step, bins=64)
        self.logger.experiment.add_histogram('Pred_Depth_Meters', y_hat_depth_meters, self.trainer.global_step, bins=64)

    def can_visualize(self):
        return (not torch.cuda.is_available() or torch.cuda.current_device() == 0) and (
                self.global_step - self.cfg.num_steps_visualization_first) % \
                self.cfg.num_steps_visualization_interval == 0

    def observer_step(self):
        if torch.cuda.is_available() and torch.cuda.current_device() != 0:
            return
        vis_transforms = self.transforms_val_test
        list_samples = []
        for i in self.cfg.observe_train_ids:
            list_samples.append(self.datasets[SPLIT_TRAIN].get(i, override_transforms=vis_transforms))
        for i in self.cfg.observe_valid_ids:
            list_samples.append(self.datasets[SPLIT_VALID].get(i, override_transforms=vis_transforms))
        list_prefix = ('train/', ) * len(self.cfg.observe_train_ids) + ('valid/', ) * len(self.cfg.observe_valid_ids)
        batch = default_collate(list_samples)
        rgb = batch[MOD_RGB]
        rgb_tags = [f'{prefix}{id}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
        with torch.no_grad():
            if torch.cuda.is_available():
                rgb = rgb.cuda()
            y_hat = self.net(rgb)
            y_hat_semseg = y_hat[MOD_SEMSEG]
            y_hat_depth = y_hat[MOD_DEPTH]
            if type(y_hat_semseg) is list:
                y_hat_semseg = y_hat_semseg[-1]
                y_hat_depth = y_hat_depth[-1]
        self.visualize(batch, y_hat_semseg, y_hat_depth, rgb_tags, 'valid/observed_samples')
