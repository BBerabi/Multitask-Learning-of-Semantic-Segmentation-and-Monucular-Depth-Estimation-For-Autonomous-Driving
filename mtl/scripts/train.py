import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from mtl.experiments.experiment_semseg_with_depth import ExperimentSemsegDepth
from mtl.utils.rules import check_all_rules, pack_submission
from mtl.utils.config import command_line_parser
from mtl.utils.daemon_tensorboard import DaemonTensorboard
from mtl.utils.daemon_ngrok import DaemonNgrok


def main():
    cfg = command_line_parser()

    check_all_rules(cfg)

    model = ExperimentSemsegDepth(cfg)

    logger = TestTubeLogger(
        save_dir=os.path.join(cfg.log_dir),
        name='tube',
        version=0,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(cfg.log_dir, 'checkpoints'),
        save_best_only=True,
        verbose=True,
        monitor='metric',
        mode='max',
        prefix=''
    )

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus='-1' if torch.cuda.is_available() else None,
        show_progress_bar=cfg.log_to_console,
        max_nb_epochs=cfg.num_epochs,
        distributed_backend='dp',   # single- or multi-gpu on a single node
        print_nan_grads=False,
        weights_summary=None,
        weights_save_path=None,
        nb_sanity_val_steps=1,
    )

    daemon_tb = None
    daemon_ngrok = None

    if not cfg.prepare_submission:
        if cfg.tensorboard_daemon_start:
            daemon_tb = DaemonTensorboard(cfg.log_dir, cfg.tensorboard_daemon_port)
            daemon_tb.start()
        if cfg.ngrok_daemon_start:
            daemon_ngrok = DaemonNgrok(cfg.ngrok_auth_token, cfg.tensorboard_daemon_port)
            daemon_ngrok.start()
        trainer.fit(model)

    # prepare submission archive with predictions, source code, training log, and the model
    dir_pred = os.path.join(cfg.log_dir, 'predictions')
    shutil.rmtree(dir_pred, ignore_errors=True)
    trainer.test(model)
    pack_submission(cfg.log_dir)

    if daemon_tb is not None:
        daemon_tb.stop()
    if daemon_ngrok is not None:
        daemon_ngrok.stop()


if __name__ == '__main__':
    main()
