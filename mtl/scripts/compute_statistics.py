import math
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtl.datasets.definitions import SPLIT_TRAIN, MOD_RGB, MOD_DEPTH
from mtl.utils.helpers import resolve_dataset_class
from mtl.utils.transforms import get_transforms


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Expecting 2 arguments: dataset name and path')
        exit(0)

    dataset_name, dataset_root = sys.argv[1], sys.argv[2]
    print(f'Computing dataset statistics of {dataset_name}')

    dataset_cls = resolve_dataset_class(dataset_name)
    ds = dataset_cls(dataset_root, SPLIT_TRAIN)

    transforms = get_transforms()
    ds.set_transforms(transforms)

    dl = DataLoader(ds, 64, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    rgb_accum, rgb_accum2, rgb_cnt = \
        torch.tensor([0,0,0], dtype=torch.float64), torch.tensor([0,0,0], dtype=torch.float64), 0.0
    depth_accum, depth_accum2, depth_cnt = 0.0, 0.0, 0.0
    depth_min, depth_max = float('inf'), -float('inf')

    for batch in tqdm(dl):
        rgb = batch[MOD_RGB].cuda()
        depth = batch[MOD_DEPTH].cuda()
        N, C, H, W = rgb.shape

        rgb = rgb.reshape(N, C, -1).permute(1, 0, 2).reshape(C, -1).double()
        rgb_accum += rgb.sum(dim=1).cpu()
        rgb_accum2 += (rgb ** 2).sum(dim=1).cpu()
        rgb_cnt += rgb.shape[-1]

        depth = depth[depth == depth].double()
        depth_accum += depth.sum().cpu().item()
        depth_accum2 += (depth ** 2).sum().cpu()
        depth_cnt += float(depth.numel())
        depth_min = min(depth_min, depth.min().item())
        depth_max = max(depth_max, depth.max().item())

    rgb_mean = rgb_accum / rgb_cnt
    rgb_mean2 = rgb_accum2 / rgb_cnt
    rgb_std = (rgb_mean2 - rgb_mean ** 2).sqrt()

    depth_mean = depth_accum / depth_cnt
    depth_mean2 = depth_accum2 / depth_cnt
    depth_std = math.sqrt(depth_mean2 - depth_mean ** 2)

    print(f'rgb mean={rgb_mean} std={rgb_std}')
    print(f'depth mean={depth_mean} std={depth_std} min={depth_min} max={depth_max}')
