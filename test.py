import os
from glob import glob
import logging
import pickle

import matplotlib.pyplot as plt

from src.dateset import LoadData


dt_dir = '/home/users/mireiffe/PycharmProjects/meeting/IntInhom/results/N2_sig1.0_dt10.0_mu1.0_nu5.0_tol0.0001_5e-05'
dt_dir = '/home/users/mireiffe/PycharmProjects/meeting/IntInhom/results/May02_N2_sig1.0_dt10.0_mu1.0_nu5.0_tol0.0001_5e-05'

ids = [os.path.splitext(file)[0]
        for file in os.listdir(dt_dir)]
logging.info(f'Creating dataset with {len(ids)} examples')

res = {}
for idx in ids:
    fname = glob(f'{os.path.join(dt_dir, idx)}*')
    assert len(fname) == 1, \
        f'Either no file or multiple files found for the ID {idx}: {fname}'
    with open(fname[0], 'rb') as f:
        data = pickle.load(f)
    res[fname[0]] = data

fig = plt.figure(100)
ax = fig.subplots(1, 1)
for k, dt in enumerate(res.items()):
    im_nm, data = dt
    ax.cla()
    ax.imshow(data['mask'])
    ax.imshow(data['img'], 'gray', alpha=0.75)
    fig.suptitle(f'iter = {k}' + im_nm)
    plt.pause(2)
