import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import Cityscapes
from tqdm import tqdm

from mtl.datasets.definitions import *


class DatasetMiniscapes(torch.utils.data.Dataset):
    """
    This is a smaller version of the SynScapes [1] dataset with RGB, Semantic, and Depth modalities.
    It provides a total of 25000 samples.
    [1]: @article{Synscapes,
             author={Magnus Wrenninge and Jonas Unger},
             title={Synscapes: A Photorealistic Synthetic Dataset for Street Scene Parsing},
             url={http://arxiv.org/abs/1810.08705},
             year={2018},
             month={Oct}
         }
    """

    def __init__(self, dataset_root, split, integrity_check=False):
        assert split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST), f'Invalid split {split}'
        self.dataset_root = dataset_root
        self.split = split
        self.transforms = None
        if integrity_check:
            for i in tqdm(range(len(self))):
                self.get(i)

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        # load rgb
        rgb = Image.open(self.get_item_path(index, MOD_RGB))
        rgb.load()
        assert rgb.mode == 'RGB'

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
        }

        # load semseg
        path_semseg = self.get_item_path(index, MOD_SEMSEG)
        if os.path.isfile(path_semseg):
            semseg = self.load_semseg(path_semseg)
            assert semseg.size == rgb.size
            out[MOD_SEMSEG] = semseg

        # load depth
        path_depth = self.get_item_path(index, MOD_DEPTH)
        if os.path.isfile(path_depth):
            depth = self.load_depth(path_depth, check_all_pixels_valid=False)
            assert depth.size == rgb.size
            out[MOD_DEPTH] = depth

        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            out = self.transforms(out)

        return out

    def get_item_path(self, index, modality):
        return os.path.join(
            self.dataset_root, self.split, modality, f'{index}.{"jpg" if modality == MOD_RGB else "png"}'
        )

    def name_from_index(self, index):
        return f'{index}'

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return {
            SPLIT_TRAIN: 20000,
            SPLIT_VALID: 2500,
            SPLIT_TEST: 2500,
        }[self.split]

    @property
    def rgb_mean(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        return [255 * 0.485, 255 * 0.456, 255 * 0.406]

    @property
    def rgb_stddev(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        return [255 * 0.229, 255 * 0.224, 255 * 0.225]

    @staticmethod
    def load_semseg(path):
        semseg = Image.open(path)
        assert semseg.mode in ('P', 'L')
        return semseg

    @staticmethod
    def save_semseg(path, img, semseg_color_map, semseg_ignore_label=None, semseg_ignore_color=(0, 0, 0)):
        if torch.is_tensor(img):
            img = img.squeeze()
            assert img.dim() == 2 and img.dtype in (torch.int, torch.long)
            img = img.cpu().byte().numpy()
            img = Image.fromarray(img, mode='P')
        palette = [0 for _ in range(256 * 3)]
        for i, rgb in enumerate(semseg_color_map):
            for c in range(3):
                palette[3 * i + c] = rgb[c]
        if semseg_ignore_label is not None:
            for c in range(3):
                palette[3 * semseg_ignore_label + c] = semseg_ignore_color[c]
        img.putpalette(palette)
        img.save(path, optimize=True)

    @property
    def semseg_num_classes(self):
        return len(self.semseg_class_names)

    @property
    def semseg_ignore_label(self):
        return 255

    @property
    def semseg_class_colors(self):
        return [clsdesc.color for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]

    @property
    def semseg_class_names(self):
        return [clsdesc.name for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]

    def depth_meters_float32_to_disparity_uint8(self, x, out_of_range_policy):
        assert out_of_range_policy in ('invalidate', 'clamp_to_range')
        x = np.array(x).astype(np.float32)
        x = 1 / x
        disparity_min, disparity_max = 1 / self.depth_meters_max, 1 / self.depth_meters_min
        x = 1 + 254 * (x - disparity_min) / (disparity_max - disparity_min)
        if out_of_range_policy == 'invalidate':
            with np.errstate(invalid='ignore'):
                x[x < 0.] = float('nan')
                x[x > 255.] = float('nan')
            x[x != x] = 0
        elif out_of_range_policy == 'clamp_to_range':
            assert np.sum((x != x).astype(np.int)) == 0
            x[x < 1.] = 1.
            x[x > 255.] = 255.
        x = x.astype(np.uint8)
        return x

    def depth_disparity_uint8_to_meters_float32(self, x, check_all_pixels_valid):
        assert type(check_all_pixels_valid) is bool
        mask_invalid = x == 0
        assert not check_all_pixels_valid or np.sum(mask_invalid.astype(np.int)) == 0
        disparity_min, disparity_max = 1 / self.depth_meters_max, 1 / self.depth_meters_min
        x = (disparity_max - disparity_min) * (x - 1).astype(np.float32) / 254 + disparity_min
        x = 1.0 / x
        x[mask_invalid] = np.nan
        return x

    def load_depth(self, path, check_all_pixels_valid):
        depth = Image.open(path)
        assert depth.mode == 'L'
        depth = np.array(depth)
        depth = self.depth_disparity_uint8_to_meters_float32(depth, check_all_pixels_valid)
        depth = Image.fromarray(depth)
        return depth

    def save_depth(self, path, img, out_of_range_policy):
        assert torch.is_tensor(img) and (img.dim() == 2 or img.dim() == 3 and img.shape[0] == 1)
        if img.dim() == 3:
            img = img.squeeze(0)
        img = img.cpu().numpy()
        img = self.depth_meters_float32_to_disparity_uint8(img, out_of_range_policy)
        img = Image.fromarray(img)
        img.save(path, optimize=True)

    @property
    def depth_meters_mean(self):
        return 27.0727

    @property
    def depth_meters_stddev(self):
        return 29.1264

    @property
    def depth_meters_min(self):
        return 4

    @property
    def depth_meters_max(self):
        return 300


if __name__ == '__main__':
    print('Checking dataset integrity...')
    ds_train = DatasetMiniscapes(sys.argv[1], SPLIT_TRAIN, integrity_check=True)
    ds_valid = DatasetMiniscapes(sys.argv[1], SPLIT_VALID, integrity_check=True)
    ds_test = DatasetMiniscapes(sys.argv[1], SPLIT_TEST, integrity_check=True)
    print('Dataset integrity check passed')
