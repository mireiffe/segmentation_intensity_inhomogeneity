import os
from glob import glob
import pickle

import logging
import numpy as np

from .functions import imset


def LoadData(dt_dir):
    '''
    make list of data
    '''
    ids = [os.path.splitext(file)[0]
            for file in os.listdir(dt_dir)]
    logging.info(f'Creating dataset with {len(ids)} examples')

    res = {}
    for idx in ids:
        fname = glob(f'{os.path.join(dt_dir, idx)}*')
        assert len(fname) == 1, \
            f'Either no file or multiple files found for the ID {idx}: {fname}'
        img = imset(load(fname[0])['img'], 'gray', 255)
        img = np.where(img == 0, 1., img)
        res[fname[0]] = img, load(fname[0])['img']
    return res


def load(filename):
    with open(filename, 'rb') as f:
        varlst = pickle.load(f)
    return varlst
