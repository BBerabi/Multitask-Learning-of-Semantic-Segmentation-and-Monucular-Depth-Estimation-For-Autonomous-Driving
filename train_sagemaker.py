#!/usr/bin/env python


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


import sys
# sys.stdout = Unbuffered(sys.stdout)
# sys.stderr = Unbuffered(sys.stderr)


import os
import traceback


exceptions_log_path = os.path.join(os.environ['SM_MODEL_DIR'], 'exceptions.log')


def custom_excepthook(exc_type, exc_value, exc_traceback):
    msg = f'type={exc_type} value={exc_value} traceback={traceback.format_tb(exc_traceback)}\n'
    with open(exceptions_log_path, 'a') as fp:
        fp.write(msg)
    print(msg, file=sys.stderr)


sys.excepthook = custom_excepthook


from torchvision import datasets

from mtl.scripts.train import main


if __name__=='__main__':
    
    # unpack dataset
    dataset_root = os.environ.get('SM_CHANNEL_TRAINING', None)
    datasets.utils.extract_archive(os.path.join(dataset_root, 'miniscapes.zip'), remove_finished=True)

    # run training
    main()
