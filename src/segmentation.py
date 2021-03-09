import os
import pickle
import time

import logging
from colorama import Fore
from tqdm import tqdm
from tqdm._utils import _term_move_up

import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools

from .functions import Functions, Operators, nzdiv


class segImg(object):
    m_eps = np.finfo(float).eps
    glb_t, glb_phi_t = 0, 0

    def __init__(self, imgs, args, initials):
        self.img = imgs[1][0]
        self.initials = initials
        self.N = args.N
        self.n_phi = int(np.ceil(np.log2(args.N)))
        self.mu = args.mu
        self.nu = args.nu
        self.tol = args.tol
        self.dt = args.dt
        self.vismode = args.vismode
        self.visterm = args.visterm

        init_time = time.strftime('%b%d', time.localtime(time.time()))
        self.sv_dir = os.path.join('results', 
                    f'{init_time}_N{self.N}_sig{args.sig}_dt{self.dt}_mu{self.mu}_nu{self.nu}_tol{self.tol[0]}_{self.tol[1]}')
        self.sv_nm = os.path.join(self.sv_dir, os.path.basename(imgs[0]))

        self.funs = Functions(eps=args.eps, m_eps=self.m_eps)
        # ksz=1: proper gradient, =3: sobel, =-1: shcarr
        self.ops = Operators(h=1, ksz=-1, m_eps=self.m_eps)

        if args.sig_scl[0]:
            dim = args.sig_scl[1] * args.sig_scl[2]
            dim_img = self.img.size
            self.sig = args.sig * np.sqrt(dim_img / dim)
        else:
            self.sig = args.sig

        self.w = np.ceil(4 * self.sig + 1) // 2 * 2 + 1
        lo = (self.w - 1) // 2
        x, y = np.arange(self.w), np.arange(self.w)
        self.ker = self.funs.fun_ker(x, y, lo, self.sig)
        self.onek = cv2.filter2D(np.ones_like(self.img), -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)

        self.err_b, self.err_c, self.err_phi = 1E5, 1E5, [1E5] * self.n_phi

    def init_vars(self):
        if self.initials is not None:
            self.phis = [cv2.resize(inis, self.img.shape[::-1], interpolation=cv2.INTER_LINEAR)
                        for inis in self.initials[0]]
            self.b = cv2.resize(self.initials[1], self.img.shape[::-1], interpolation=cv2.INTER_LINEAR)
            self.c = self.initials[2]

        else:
            self.b = np.ones_like(self.img)
            self.c = [0] * self.N

            _mu, _sig = self.img.mean(), self.img.std()
            _lev = range(1, 1 - self.n_phi, -1)
            _lev = range(1, 1 + self.n_phi)
            self.phis = [np.where(self.img > _mu + l * _sig, -1., 1.) for l in _lev]

    def updt_bbk(self):
        self.bk = cv2.filter2D(self.b, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        self.bbk = cv2.filter2D(self.b * self.b, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        
    def updt_bc(self):
        _oldc, _oldb = np.copy(self.b), np.copy(self.c)
        # update c
        self.H = [self.funs.hvsd(phi) for phi in self.phis]
        H_ref = [(h, 1 - h) for h in self.H]
        M = [np.prod(m, axis=0) for m in itertools.product(*H_ref)]
        c_den = [np.sum(self.bk * self.img * m) for m in M]
        c_num = [np.sum(self.bbk * m) for m in M]
        self.c = nzdiv(c_den, c_num)

        # update b
        J1 = np.sum(np.prod((self.c, M), axis=0))
        J2 = np.sum(np.prod((np.power(self.c, 2), M), axis=0))
        b_den = cv2.filter2D(self.img * J1, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        b_num = cv2.filter2D(J2, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        self.b = nzdiv(b_den, b_num)
        _newc, _newb = np.copy(self.b), np.copy(self.c)
        self.err_b = nzdiv(self.ops.norm(_oldb - _newb), (self.ops.norm(_newb)))
        self.err_c = nzdiv(self.ops.norm(_oldc - _newc), (self.ops.norm(_newc)))

        self.err_b /= self.funs.fun_dt(self.glb_t) * self.dt
        self.err_c /= self.funs.fun_dt(self.glb_t) * self.dt

    def updt_phi(self):
        self.updt_bbk()
        _e = [
            self.img ** 2 * self.onek - 2 * ci * self.img * self.bk + ci ** 2 * self.bbk
            for ci in self.c
        ]
        phi_t = 0

        _pbar = tqdm(
            total=50,
            desc=f'Updating phi', unit='iter', leave=False, 
            bar_format='{l_bar}%s{bar:25}%s{r_bar}{bar:-25b}' % (Fore.BLUE, Fore.RESET)
        )
        while True:
            phi_t += 1
            self.glb_phi_t += 1

            _old = np.copy(self.phis)

            gphis = [self.ops.grad_img(phi) for phi in self.phis]
            N_gphis = [self.ops.norm(gphi) for gphi in gphis]
            
            kappa = [self.ops.cvt_phi(phi, ksz=1) for phi in self.phis]
            delta_phi = [self.funs.delta(phi) for phi in self.phis]

            dp_dpgs = [
                [self.funs.fun_dp(n_gphi) * gphi[0], self.funs.fun_dp(n_gphi) * gphi[1]]
                for n_gphi, gphi in list(zip(*[N_gphis, gphis]))
            ]
            divg = [self.ops.div_phi(ddp, ksz=1) for ddp in dp_dpgs]

            if self.N >= 1 and self.N <= 2:
                dE = [_e[0] - _e[1]]
            elif self.N >= 3 and self.N <= 4:
                dE = [self.H[1] * (_e[0] - _e[2]) + (1 - self.H[1]) * (_e[1] - _e[3]),
                    self.H[0] * (_e[0] - _e[1]) + (1 - self.H[0]) * (_e[2] - _e[3])]
            else:
                print('Use appropriate value of N!!')

            dphis = [
                - dp * de + self.nu * dp * kp + self.mu * dv
                for dp, kp, dv, de in list(zip(*[delta_phi, kappa, divg, dE]))
            ]
            self.phis = [
                phi + (self.dt * self.funs.fun_dt(self.glb_t)) * nzdiv(dphi, np.abs(dphi).max())
                for phi, dphi in list(zip(*[self.phis, dphis]))
            ]
            if phi_t == 1:
                self.phis = [np.where(phi < 0, -1., 1.) for phi in self.phis]
            
            _new = np.copy(self.phis)
            err_reg = np.where(np.abs(_new) < 1.5, 1., 0.)

            self.err_phi = [
                self.ops.norm(err_reg * (o - n)) / err_reg.sum() / self.dt / self.funs.fun_dt(self.glb_t)
                for o, n in list(zip(*[_old, _new]))
            ]

            _pbar.set_postfix_str('Error phi='+', '.join([f'{ep:.2E}' for ep in self.err_phi]))
            _pbar.update()
            
            self.guis()

            if (phi_t > 10 and np.max(self.err_phi) < self.tol[0]) or phi_t > 50:
                _pbar.close()
                break

    def guis(self, fignum=500, keep=False, keep_time=0.01, mask=None):
        if self.vismode:
            if self.visterm == 0:
                if mask is not None:
                    fig = plt.figure(fignum)
                    ax = fig.subplots(2, 1)
                    ax[0].cla()
                    ax[0].imshow(self.img, 'gray')
                    clrs = ['red', 'green']
                    for i, phi in enumerate(self.phis):
                        ax[0].contour(phi, levels=0, colors=clrs[i], linestyles='solid')
                    ax[1].imshow(mask)
                    ax[1].imshow(self.img, 'gray', alpha=0.75)
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)
            elif self.glb_phi_t % self.visterm == 0 or mask is not None:
                fig = plt.figure(fignum)
                if mask is not None:
                    fig.clf()
                    ax = fig.subplots(2, 1)
                    ax[0].cla()
                    ax[0].imshow(self.img, 'gray')
                    clrs = ['red', 'green']
                    for i, phi in enumerate(self.phis):
                        ax[0].contour(phi, levels=0, colors=clrs[i], linestyles='solid')
                    ax[1].imshow(mask)
                    ax[1].imshow(self.img, 'gray', alpha=0.75)
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)
                else:
                    fig.clf()
                    ax = fig.subplots(1, 1)
                    ax.cla()
                    ax.imshow(self.img, 'gray')
                    clrs = ['red', 'green']
                    for i, phi in enumerate(self.phis):
                        ax.contour(phi, levels=0, colors=clrs[i], linestyles='solid')
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)

    def saving(self, vardict):
        try:
            os.mkdir(self.sv_dir)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        with open(self.sv_nm, 'wb') as f:
            pickle.dump(vardict, f)
        logging.info(f'{self.sv_nm} is saved!!')