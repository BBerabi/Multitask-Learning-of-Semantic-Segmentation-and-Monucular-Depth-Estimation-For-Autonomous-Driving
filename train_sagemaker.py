#!/usr/bin/env python
import os

from torchvision import datasets

from mtl.scripts.train import main

if __name__=='__main__':
    
    # unpack dataset
    dataset_root = os.environ.get('SM_CHANNEL_TRAINING', None)
    datasets.utils.extract_archive(os.path.join(dataset_root, 'miniscapes.zip'), remove_finished=True)

    # run training
    main()
