'''
my tools
'''
import numpy as np
import cv2
from tqdm import tqdm
from tqdm.utils import _term_move_up
from colorama import Fore
from scipy.ndimage import binary_dilation
from skimage.segmentation import flood_fill

from .functions import nzdiv


def labeling(img, dist=1): # background: 0 / foreground: 1
    res = -1 * img

    node = []
    ind = []
    bins = []
    sz = []
    k = 0

    pbar = tqdm(
        total=len((res == -1).nonzero()[0]),
        desc=f'Region labeling...', unit='iter', leave=True,
        bar_format='{l_bar}%s{bar:25}%s{r_bar}{bar:-25b}' % (Fore.GREEN, Fore.RESET)
    )
    pbar.write(_term_move_up() * 2, end='\r')
    while True:
        y, x = (res == -1).nonzero()
        if len(y) == 0:
            break
        k += 1
        res = flood_fill(res, (y[0], x[0]), k, connectivity=dist)
        node += [(y[0], x[0])]
        ind += [np.where(res == k)]
        bins += [np.where(res == k, 1., 0.)]
        sz += [len(ind[k - 1][0])]
        pbar.update(sz[-1])
    pbar.close()
    return {'indices': ind, 'sizes': sz, 'nodes': node, 'number_region': k,
            'binary_maps': bins, 'colored': res, 'img': img}


def reg_del(img, omg, c, dil_sz, shr_sz, iter_num=1):
    m, n = img.shape
    dil = int(np.sqrt(m * n) / 3)
    c = c[0]
    omg = omg[0]

    reg = labeling(omg, 2)
    mu = [(img[idx].sum() / len(idx[0])) for idx in reg['indices']]
    mu_img = np.ones_like(omg)
    for i, idx in enumerate(reg['indices']): mu_img[idx] = mu[i] 

    loc_mu = []
    pbar = tqdm(
        total=reg['number_region'],
        desc=f'Normalizing intnensities...', unit='iter', leave=True,
        bar_format='{l_bar}%s{bar:25}%s{r_bar}{bar:-25b}' % (Fore.BLUE, Fore.RESET)
    )
    pbar.write(_term_move_up(), end='\r')
    loc_mu_img = np.zeros_like(omg)
    for idx in reg['indices']:
        bn = np.zeros_like(omg)
        bn[idx] = 1
        dil_bn = cv2.filter2D(bn, -1, np.ones((2 * dil + 1, 2 * dil + 1))) > .5
        mu_dil_bn = np.where(dil_bn > .5, mu_img, 0.)
        loc_mu += [mu_dil_bn.sum() / dil_bn.sum()]
        loc_mu_img[idx] = loc_mu[-1]
        pbar.update()
    pbar.close()
    nml_img = nzdiv(omg * img, loc_mu_img)

    nml_mu = [nzdiv(nml_img.sum(), omg.sum())]
    del_omg = [np.where(nml_img > nml_mu[0], 1., 0.)]
    for k in range(iter_num - 1):
        nml_mu += [nzdiv((del_omg[k] * nml_img).sum(), del_omg[k].sum())]
        del_omg += [np.where(del_omg[k] * nml_img > nml_mu[k + 1], 1., 0.)] 

    dd_omg = cv2.filter2D(del_omg[-1], -1, np.ones((2 * dil_sz + 1, 2 * dil_sz + 1))) > .5
    sdd_omg = cv2.filter2D((1 - dd_omg).astype('uint8'), -1, np.ones((2 * shr_sz + 1, 2 * shr_sz + 1)))
    mask = np.where(sdd_omg == 0, 1., 0.)
    return mask