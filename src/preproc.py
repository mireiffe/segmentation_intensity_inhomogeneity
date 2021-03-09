import cv2
import numpy as np


def shrink_imgs(imgs, tgsz):
    tgdim = tgsz[0] * tgsz[1]
    m, n = imgs[1][0].shape[:2]
    rat = m / n
    new_y, new_x = int(np.sqrt(tgdim * rat)), int(np.sqrt(tgdim / rat))
    img = cv2.resize(imgs[1][0], (new_x, new_y), interpolation=cv2.INTER_LINEAR)
    ori_img = cv2.resize(img[1][1], (new_x, new_y), interpolation=cv2.INTER_LINEAR)
    return [imgs[0], (img, ori_img)]