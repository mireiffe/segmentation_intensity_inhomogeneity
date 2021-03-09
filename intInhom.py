'''
model based reflection detection
'''
import os
import sys
from time import time

import pickle
import logging
from tqdm.auto import tqdm
from tqdm.utils import _term_move_up
import argparse
from colorama import Fore

import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt

from src.functions import nzdiv
from src.segmentation import segImg
from src.dateset import LoadData
from src.postproc import reg_del
from src.preproc import shrink_imgs


def main(imgs, args, save_var=True, initials=None, max_iter=500):
    ori_img = imgs[1][1]
    seg = segImg(imgs, args, initials=initials)
    logging.info(f'''Information:
                Number of reg.: {seg.N}
                Number of phi:  {seg.n_phi}
                Sigma(orig):    {args.sig}
                Sigma:          {seg.sig}
                Window size:    {seg.w}
                Curv. coeff.:   {seg.nu}
                Div. coeff.:    {seg.mu}
                Time increment: {seg.dt}
                Tol phi:        {seg.tol[0]}
                Tol b & c:      {seg.tol[1]}
                Vismode:        {args.vismode}
                Visterm:        {args.visterm}
                Save directory: {seg.sv_nm}
        ''')
    
    seg.init_vars()
    seg.updt_bbk()
    seg.updt_bc()

    pbar = tqdm(
        total=int(seg.err_c / seg.tol[1] * seg.glb_t),
        desc=f'Global progress', unit='iter', leave=False,
        bar_format='{l_bar}%s{bar:25}%s{r_bar}{bar:-25b}' % (Fore.RED, Fore.RESET)
    )
    while True:
        seg.glb_t += 1
        pbar.set_postfix_str(f'dt={seg.dt * seg.funs.fun_dt(seg.glb_t):.2f}, '
                            f'c={seg.c}, '
                            f'Error b={seg.err_b:.2E}, '
                            f'Error c={seg.err_c:.2E}')
        pbar.total = min(int(seg.err_c / seg.tol[1] * seg.glb_t), max_iter)
        pbar.write(_term_move_up() * 2, end='\r')
        pbar.update()

        seg.updt_phi()
        seg.updt_bc()

        if seg.err_c < args.tol[1] or seg.glb_t >= max_iter:
            idct = [phi > 0 for phi in seg.phis]
            idct_ref = [(h, 1 - h) for h in idct]
            omg = [np.prod(m, axis=0) for m in itertools.product(*idct_ref)]

            J = np.sum([ci * og for ci, og in list(zip(*[seg.c, omg]))], axis=0)

            agsrt_c = np.argsort(seg.c)[::-1]
            srt_c = [seg.c[i] for i in agsrt_c]
            srt_omg = [omg[i] for i in agsrt_c]
            pbar.close()
            if save_var:
                mask = reg_del(seg.img, srt_omg, srt_c, 4, 2, iter_num=2)
                seg.guis(keep_time=10, mask=mask)
                vardict = {
                    'ori_img': ori_img, 'img': seg.img,
                    'phi': seg.phis, 'b': seg.b, 'c': seg.c, 'iters': [seg.glb_t, seg.glb_t], 
                    'mask': mask, 'J': J, 'srt_c': srt_c, 'srt_omg': srt_omg
                }
                # seg.saving(vardict)
                break
            return (seg.phis, seg.b, seg.c)


def get_args():
    parser = argparse.ArgumentParser(description='Extracting specular reflection of teeth image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-N', metavar='N', type=int, default=2,
                        help='Number of regions to segmented', dest='N')
    parser.add_argument('-sig', metavar='SIGMA', type=float, default=1,
                        help='Sigma for gaussian kernel', dest='sig')
    parser.add_argument('-mu', metavar='MU', type=float, default=1,
                        help='Coefficient for divergence term', dest='mu')
    parser.add_argument('-nu', metavar='NU', type=float, default=5,
                        help='Coefficient for the curvature term', dest='nu')
    parser.add_argument('-tol', metavar='TOL', nargs='+', type=float, default=[1E-4, 5E-5],
                        help='Tolerences for phi and b & c', dest='tol')
    parser.add_argument('-dt', metavar='DT', type=float, default=5,
                        help='Time increment', dest='dt')
    parser.add_argument('-eps', metavar='EPSILON', type=float, default=1,
                        help='Epsilon', dest='eps')
    parser.add_argument('--sig_scl', metavar='SIGSCL', type=eval, nargs='+', default=[True, 512, 512],
                        help='Scaling sigma or not', dest='sig_scl')
    parser.add_argument('--vismode', metavar='VISMODE', type=eval, default=False,
                        help='Turn on the visualization', dest='vismode')
    parser.add_argument('--visterm', metavar='VISTERM', type=int, default=5,
                        help='Term for visualization', dest='visterm')
    parser.add_argument('--dt_dir', dest='dt_dir', type=str, default=None,
                        help='Database directory')
    parser.add_argument('--range', metavar='RANGE', nargs='+', type=int, default=[-1, 10000],
                        help='range of img', dest='range')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rseed = 910930
    np.random.seed(rseed)
    args = get_args()

    img_data = LoadData(args.dt_dir)

    for imgs in img_data.items():
        im_num = int(os.path.basename(imgs[0])[1:4])
        if im_num < args.range[0] or im_num > args.range[1]:
            continue
        try:
            parameters = main(shrink_imgs(imgs, args.sig_scl[1:]), args, save_var=False, max_iter=100)
            main(imgs, args, initials=parameters)
            # main(imgs, args)
        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
