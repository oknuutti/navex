import numpy as np
import matplotlib.pyplot as plt
import cv2


def unit_aflow(W, H):
    return np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), axis=2)


def save_aflow(fname, aflow):
    if fname[-4:].lower() != '.png':
        fname = fname + '.png'
    aflow_int = np.clip(aflow * 8 + 0.5, 0, 2**16 - 1).astype('uint16')
    aflow_int[np.isnan(aflow)] = 2**16 - 1
    aflow_int = np.concatenate((aflow_int, np.zeros((*aflow_int.shape[:2], 1), dtype='uint16')), axis=2)
    cv2.imwrite(fname, aflow_int, (cv2.IMWRITE_PNG_COMPRESSION, 9))


def load_aflow(fname):
    aflow = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.float32)
    aflow[np.isclose(aflow, 2**16 - 1)] = np.nan
    return aflow / 8


def show_pair(img1, img2, aflow, pts=8):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.array(img1))
    axs[1].imshow(np.array(img2))
    for i in range(pts):
        idx = np.argmax(np.logical_not(np.isnan(aflow[:, :, 0])).flatten().astype(np.float32)
                        * np.random.lognormal(0, 1, (np.prod(aflow.shape[:2]),)))
        y0, x0 = np.unravel_index(idx, aflow.shape[:2])
        axs[0].plot(x0, y0, 'x')
        axs[1].plot(aflow[y0, x0, 0], aflow[y0, x0, 1], 'x')

    plt.show()
