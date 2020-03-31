import difflib
import os
import zipfile

from mtl.datasets.definitions import MOD_SEMSEG, MOD_DEPTH
from mtl.utils.config import EXPERIMENT_INVARIANT_KEYS


def add_filetree_to_zip(zip, dir_src, filter_filename=None, filter_dirname=None):
    dir_src = os.path.abspath(dir_src)
    dir_src_name = os.path.basename(dir_src)
    dir_src_parent_dir = os.path.dirname(dir_src)
    zip.write(dir_src, arcname=dir_src_name)
    for cur_dir, _, cur_filenames in os.walk(dir_src):
        if filter_dirname is not None and filter_dirname(os.path.basename(cur_dir)):
            continue
        if cur_dir != dir_src:
            zip.write(cur_dir, arcname=os.path.relpath(cur_dir, dir_src_parent_dir))
        for filename in cur_filenames:
            if filter_filename is not None and filter_filename(filename):
                continue
            zip.write(
                os.path.join(cur_dir, filename),
                arcname=os.path.join(os.path.relpath(cur_dir, dir_src_parent_dir), filename)
            )


def pack_source_dir(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    cfg_str = '\n'.join([f'{k}: {v}' for k, v in cfg.__dict__.items() if k not in EXPERIMENT_INVARIANT_KEYS])
    with zipfile.ZipFile(path_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(
            zip,
            dir_src,
            filter_filename=lambda f: not f.endswith('.py'),
            filter_dirname=lambda d: d in ('__pycache__',),
        )
        zip.writestr('cmd.txt', cfg_str)


def diff_source_dir_and_zip(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    with zipfile.ZipFile(path_zip) as zip:
        for file in zip.namelist():
            if file == 'cmd.txt':
                continue
            file_info = zip.getinfo(file)
            if file_info.is_dir():
                continue
            path_src = os.path.join(os.path.dirname(dir_src), file)
            if not os.path.isfile(path_src):
                raise FileNotFoundError(path_src)
            with open(path_src) as f:
                lines_src = f.read().split('\n')
            lines_zip = zip.read(file).decode('utf-8').split('\n')
            lines_diff = list(difflib.unified_diff(lines_zip, lines_src))
            if len(lines_diff) > 0:
                raise Exception(
                    f'File "{file}" changed, check README for the recommended workflow:\n' +
                    f'\n'.join(lines_diff)
                )
        cfg_src = [f'{k}: {v}' for k, v in cfg.__dict__.items() if k not in EXPERIMENT_INVARIANT_KEYS]
        cfg_zip = zip.read('cmd.txt').decode('utf-8').split('\n')
        cfg_diff = list(difflib.unified_diff(cfg_zip, cfg_src))
        if len(cfg_diff) > 0:
            raise Exception(
                f'Command line changed, check README for the recommended workflow:\n' +
                f'\n'.join(cfg_diff)
            )


def pack_submission(log_dir):
    dir_pred = os.path.join(log_dir, 'predictions')
    dir_checkpoints = os.path.join(log_dir, 'checkpoints')
    path_source = os.path.join(log_dir, 'source.zip')
    path_metrics = os.path.join(log_dir, 'tube', 'version_0', 'metrics.csv')
    assert os.path.isdir(dir_pred), f'Predictions directory is missing at {dir_pred}'
    assert os.path.isdir(dir_checkpoints), \
        f'Model checkpoints directory is missing at {dir_checkpoints}; was the model trained for at least one epoch?'
    assert os.path.isfile(path_source), \
        f'Source archive is missing at {path_source}, without it submission will not be accepted'
    assert os.path.isfile(path_metrics), \
        f'Tensorboard metrics file is missing at {path_metrics}, without it submission will not be accepted'
    with zipfile.ZipFile(os.path.join(log_dir, 'submission.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(
            zip,
            dir_pred,
            lambda f: not f.endswith('.png'),
            lambda d: d not in (MOD_SEMSEG, MOD_DEPTH)
        )
        add_filetree_to_zip(zip, dir_checkpoints, filter_filename=lambda f: not f.endswith('.ckpt'))
        zip.write(path_source, arcname='source.zip')
        zip.write(path_metrics, arcname='metrics.csv')


def check_all_rules(cfg):
    dir_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    common_repo_log = os.path.abspath(os.path.commonprefix([dir_repo, cfg.log_dir]))
    common_repo_dataset = os.path.abspath(os.path.commonprefix([dir_repo, cfg.dataset_root]))
    path_source_zip = os.path.join(cfg.log_dir, 'source.zip')
    assert dir_repo != common_repo_log, 'Log directory must be outside of the code directory'
    assert dir_repo != common_repo_dataset, 'Dataset must be outside of the code directory'
    assert not cfg.prepare_submission or os.path.isdir(cfg.log_dir), \
        'Prior to preparing a submission, one needs to train the model first'
    assert not os.path.isdir(cfg.log_dir) or os.path.isfile(path_source_zip), \
        'Log directory exists, but "source.zip" was not found in it. Either put it back or remove the log ' \
        'directory and start the experiment again. Check README for the recommended workflow'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        pack_source_dir(cfg, dir_repo, path_source_zip)
    else:
        diff_source_dir_and_zip(cfg, dir_repo, path_source_zip)
