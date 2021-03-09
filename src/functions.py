'''
functions
'''
import numpy as np
import cv2


def imset(img, opt='multi', maxinsty=1):
    if type(img) is not np.ndarray:
        print("The input image type should be numpy.ndarray")
        return -1
    if opt == 'gray':
        if img.ndim >= 3:
            img = img.mean(axis=2)
    else:
        if img.ndim < 3:
            print("Dimension of the input variable is wrong!")
    return nzdiv(img, img.max()) * maxinsty


def nzdiv(a, b, lim=1E-05):
    if np.isscalar(a) and np.isscalar(b):
        if abs(b) <= lim:
            return a
        return a / b
    else:
        return np.divide(a, b, where=np.abs(b) > lim,
                         out=a * np.ones_like(b))


class Functions(object):
    def __init__(self, eps, m_eps):
        self.eps = eps
        self.m_eps = m_eps

    def hvsd(self, x):
        return .5 * (1 + 2 / np.pi * np.arctan(x / self.eps))

    def delta(self, x):
        return self.eps / np.pi / (self.eps ** 2 + x ** 2)

    def fun_p(self, x):
        return np.where(x <= 1, (1 - np.cos(2 * np.pi * x)) / (2 * np.pi) ** 2, (x - 1) ** 2 / 2)

    def fun_dp(self, x):
        res = np.where(x <= 1, np.sin(2 * np.pi * x) / 2 / np.pi, x - 1)
        return nzdiv(res, x)

    def fun_ker(self, x, y, lo, sig):
        res = np.exp(-((x[None, :] - lo) ** 2 + (y[:, None] - lo) ** 2) / (2 * sig) ** 2)
        return res / res.sum()

    def fun_dt(self, k):
        prd = 10
        min_lev = 0.01
        a, b = .5 + min_lev / 2, .5 - min_lev / 2
        return  b * (np.cos(2 * np.pi / prd * k) + 1) * prd / (k + prd) + min_lev

class Operators(object):
    def __init__(self, h, ksz, m_eps):
        self.h = h
        self.ksz = ksz
        self.m_eps = m_eps

    def diff_img(self, img, dx, dy, ksz=None):
        _ksz = ksz if ksz is not None else self.ksz
        return cv2.Sobel(img, -1, dx, dy, ksize=_ksz)

    def grad_img(self, img, ksz=None):
        _ksz = ksz if ksz is not None else self.ksz
        res = (
            self.diff_img(img, 1, 0, ksz=_ksz) / (2 * self.h),
            self.diff_img(img, 0, 1, ksz=_ksz) / (2 * self.h)
        )
        return res

    def cvt_phi(self, phi, ksz=None):
        _ksz = ksz if ksz is not None else self.ksz
        x, y = self.grad_img(phi, _ksz)
        xx = self.diff_img(phi, 2, 0, _ksz) / self.h ** 2
        yy = self.diff_img(phi, 0, 2, _ksz) / self.h ** 2
        xy = self.diff_img(phi, 1, 1, _ksz) / self.h ** 2
        res_den = xx * y * y - 2 * x * y * xy + yy * x * x
        res_num = np.power(x ** 2 + y ** 2, 1.5)
        return nzdiv(res_den, res_num)

    def div_phi(self, v, ksz=None):
        _ksz = ksz if ksz is not None else self.ksz
        res = self.diff_img(v[0], 1, 0, _ksz) + self.diff_img(v[1], 0, 1, _ksz)
        return res / (2 * self.h)

    def norm(self, v):
        if isinstance(v, np.ndarray):
            return np.sqrt((v ** 2).sum())
        else:
            res = 0
            for i in v:
                res += i ** 2
            return np.sqrt(res)
