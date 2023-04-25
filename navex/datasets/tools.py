import math
import os
import re
import random
import sqlite3
from re import Pattern
from typing import Tuple, Union, List, Iterable

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.interpolate import NearestNDInterpolator


def _find_files_recurse(root, path, samples, npy, ext, test, depth, relative):
    if isinstance(ext, Pattern):
        expat = ext
    else:
        if npy:
            ext = ('.npy',)
        elif isinstance(ext, str):
            ext = (ext,)
        expat = '(' + '|'.join(map(re.escape, ext)) + ')$'

    for fname in os.listdir(os.path.join(root, path)):
        fullpath = os.path.join(root, path, fname)
        if re.search(expat, fname):
            ok = test is None
            if not ok:
                try:
                    ok = test(fullpath)
                except Exception as e:
                    print('%s' % e)
            if ok:
                samples.append(os.path.join(path, fname) if relative else fullpath)
            elif 0:
                print('rejected: %s' % fullpath)    # use to debug
        elif depth > 0 and os.path.isdir(fullpath):
            _find_files_recurse(root, os.path.join(path, fname), samples, npy, ext, test, depth-1, relative)


def find_files_recurse(root, npy=False, ext='.jpg', test=None, depth=100, relative=False):
    samples = []
    _find_files_recurse(root, '', samples, npy, ext, test, depth, relative)
    samples = sorted(samples)
    return samples


def find_files(root, npy=False, ext='.jpg', test=None, relative=False):
    return find_files_recurse(root, npy, ext, test, 0, relative)


def unit_aflow(W, H):
    return np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), axis=2)


def save_aflow(fname, aflow):
    if fname[-4:].lower() != '.png':
        fname = fname + '.png'
    aflow_int = np.clip(aflow * 8 + 0.5, 0, 2**16 - 1).astype('uint16')
    aflow_int[np.isnan(aflow)] = 2**16 - 1
    aflow_int = np.concatenate((aflow_int, np.zeros((*aflow_int.shape[:2], 1), dtype='uint16')), axis=2)
    cv2.imwrite(fname, aflow_int, (cv2.IMWRITE_PNG_COMPRESSION, 9))


def load_aflow(fname, img1_size=None, img2_size=None):
    aflow = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if aflow is None:
        if os.path.exists(fname):
            raise IOError('Could not load aflow from file %s' % fname)
        else:
            raise FileNotFoundError('Could not find aflow file %s' % fname)
    h, w, _ = aflow.shape
    aflow = aflow[:, :, :2].reshape((-1, 2)).astype(np.float32)
    aflow[aflow[:, 0] >= 2**16 - 2, :] = np.nan
    return aflow.reshape((h, w, 2)) / 8


def save_xyz(path, xyz, as_png=True):
    return save_float_img(path, xyz, as_png)


def load_xyz(path):
    return load_float_img(path, 3)


def save_mono(path, x, as_png=True):
    return save_float_img(path, x, as_png)


def load_mono(path):
    return load_float_img(path, 1)


def save_float_img(path, data, as_png=True):
    data = np.atleast_3d(data)
    assert len(data.shape) == 3 and data.shape[2] in (1, 3, 4), "data must be 1, 3 or 4 channel image"

    if as_png:
        isnan, isinf = np.isnan(data), np.isinf(data)
        data[isinf] = np.nan

        mins = np.nanmin(data, axis=(0, 1))
        maxs = np.nanmax(data, axis=(0, 1))
        scs = list(map(lambda v: (2 ** 16 - 3) / (v[1] - v[0]), zip(mins, maxs)))
        mins, scs = mins.reshape((1, 1, -1)), np.array(scs).reshape((1, 1, -1))
        sc_data = np.clip((data.astype(np.float32) - mins) * scs, 0, 2 ** 16 - 3).astype(np.uint16)
        sc_data[isnan] = 2 ** 16 - 2
        sc_data[isinf] = 2 ** 16 - 1

        meta = mins.astype(np.float32).tobytes() + scs.astype(np.float32).tobytes()
        ok, img = cv2.imencode('.png', sc_data, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        if not ok:
            return ok

        with open(path, 'wb') as fh:
            fh.write(meta)
            fh.write(img)

    else:
        ok = cv2.imwrite(path + '.exr', data, (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))

    return ok


def load_float_img(path, channels):
    is_png = path[-4:] != '.exr'
    if is_png:
        with open(path, 'rb') as fh:
            bytes = fh.read()
        meta_len = 4 * 2 * channels
        meta = np.frombuffer(bytes[:meta_len], dtype=np.float32)
        mins = np.array(meta[:channels]).reshape((1, 1, -1))
        scs = np.array(meta[channels:]).reshape((1, 1, -1))
        sc_data = cv2.imdecode(np.frombuffer(bytes[meta_len:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        x = sc_data.astype(np.float32) / scs + mins
        x[sc_data == 2 ** 16 - 2] = np.nan
        x[sc_data == 2 ** 16 - 1] = np.inf
    else:
        x = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    return x


def load_mono(path):
    is_png = path[-4:] != '.exr'
    if is_png:
        with open(path, 'rb') as fh:
            bytes = fh.read()
        meta_len = 4 * 2
        minx, scx = np.frombuffer(bytes[:meta_len], dtype=np.float32)
        scaled_x = cv2.imdecode(np.frombuffer(bytes[meta_len:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        x = scaled_x.astype(np.float32) / scx + minx
        x[scaled_x == 2 ** 16 - 2] = np.nan
        x[scaled_x == 2 ** 16 - 1] = np.inf
    else:
        x = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    return x


def show_pair(img1, img2, aflow, file1='', file2='', afile='', pts=8, axs=None, show=True):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(np.array(img1))
    axs[1].imshow(np.array(img2))
    for i in range(pts):
        idx = np.argmax(np.logical_not(np.isnan(aflow[:, :, 0])).flatten().astype(np.float32)
                        * np.random.lognormal(0, 1, (np.prod(aflow.shape[:2]),)))
        y0, x0 = np.unravel_index(idx, aflow.shape[:2])
        axs[0].plot(x0, y0, 'x')
        axs[0].set_title(file1)
        axs[1].plot(aflow[y0, x0, 0], aflow[y0, x0, 1], 'x')
        axs[1].set_title(file2)

    if afile:
        plt.gcf().suptitle(afile)

    if show:
        plt.tight_layout()
        plt.show()


def valid_asteriod_area(img, min_intensity=50, remove_limb=True):
    img = np.array(img)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    _, mask = cv2.threshold(img, min_intensity, 255, cv2.THRESH_BINARY)
    r = min(*img.shape) // 40
    d = r*2 + 1
    kernel = cv2.circle(np.zeros((d, d), dtype=np.uint8), (r, r), r, 255, -1)
    star_kernel = cv2.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 4, 255, -1)

    # exclude asteroid limb from feature detection
    mask = cv2.erode(mask, star_kernel, iterations=1)   # remove stars
    mask = cv2.dilate(mask, kernel, iterations=1)       # remove small shadows inside asteroid
    mask = cv2.erode(mask, kernel, iterations=2 if remove_limb else 1)  # remove dilation and possibly asteroid limb

    return mask


def normalize_v(v):
    l = np.linalg.norm(v)
    return v if l == 0 else v/l


def normalize_mx(mx, axis=1):
    norm = np.linalg.norm(mx, axis=axis)
    with np.errstate(invalid='ignore'):
        mx /= norm[:, None]
        mx = np.nan_to_num(mx)
    return mx


def ypr_to_q(dec, ra, cna):
    if dec is None or ra is None or cna is None:
        return None

    # intrinsic euler rotations z-y'-x'', first right ascencion, then declination, and last celestial north angle
    return (
            np.quaternion(math.cos(ra / 2), 0, 0, math.sin(ra / 2))
            * np.quaternion(math.cos(-dec / 2), 0, math.sin(-dec / 2), 0)
            * np.quaternion(math.cos(-cna / 2), math.sin(-cna / 2), 0, 0)
    )


def from_opencv_q(q):
    """
    Convert an orientation from opencv convention where: cam axis +z, up -y to convention where: cam axis +x, up +z
    """
    sc2cv_q = np.quaternion(0.5, -0.5, 0.5, -0.5)
    return sc2cv_q * q * sc2cv_q.conj()


def from_opencv_v(v):
    """
    Convert an orientation from opencv convention where: cam axis +z, up -y to convention where: cam axis +x, up +z
    """
    sc2cv_q = np.quaternion(0.5, -0.5, 0.5, -0.5)
    return q_times_v(sc2cv_q, v)


def to_opencv_mx(mx):
    """
    Convert an orientation from opencv convention where: cam axis +z, up -y to convention where: cam axis +x, up +z
    """
    sc2cv_q = np.quaternion(0.5, -0.5, 0.5, -0.5)
    return q_times_mx(sc2cv_q.conj(), mx)


def q_to_ypr(q):
    # from https://math.stackexchange.com/questions/687964/getting-euler-tait-bryan-angles-from-quaternion-representation
    q0, q1, q2, q3 = quaternion.as_float_array(q)
    roll = np.arctan2(q2 * q3 + q0 * q1, .5 - q1 ** 2 - q2 ** 2)
    pitch = np.arcsin(np.clip(-2 * (q1 * q3 - q0 * q2), -1, 1))
    yaw = np.arctan2(q1 * q2 + q0 * q3, .5 - q2 ** 2 - q3 ** 2)
    return yaw, pitch, roll


def eul_to_q(angles, order='xyz', reverse=False):
    assert len(angles) == len(order), 'len(angles) != len(order)'
    q = quaternion.one
    idx = {'x': 0, 'y': 1, 'z': 2}
    for angle, axis in zip(angles, order):
        w = math.cos(angle / 2)
        v = [0, 0, 0]
        v[idx[axis]] = math.sin(angle / 2)
        dq = np.quaternion(w, *v)
        q = (dq * q) if reverse else (q * dq)
    return q


def q_times_v(q, v):
    qv = np.quaternion(0, *v)
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])


def q_times_mx(q, mx):
    qqmx = q * mx2qmx(mx) * q.conj()
    aqqmx = quaternion.as_float_array(qqmx)
    return aqqmx[:, 1:].astype(mx.dtype)


def mx2qmx(mx):
    qmx = np.zeros((mx.shape[0], 4))
    qmx[:, 1:] = mx
    return quaternion.as_quat_array(qmx)


def spherical2cartesian(lat, lon, r):
    theta = math.pi / 2 - lat
    phi = lon
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


def cartesian2spherical(x, y, z):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    lat = math.pi / 2 - theta
    lon = phi
    return np.array([lat, lon, r])


def tf_view_unit_v(sc_trg_q):
    # assumes that cam axis +x, up +z
    return q_times_v(sc_trg_q, np.array([-1, 0, 0]))


def plot_vectors(pts3d, scatter=True, conseq=True, neg_z=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    if scatter:
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    else:
        if conseq:
            ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(1, 0, len(pts3d))))
        for i, v1 in enumerate(pts3d):
            if v1 is not None:
                ax.plot((0, v1[0]), (0, v1[1]), (0, v1[2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    if neg_z:
        ax.view_init(90, -90)
    else:
        ax.view_init(-90, -90)
    plt.show()


def vector_projection(a, b):
    return a.dot(b) / b.dot(b) * b


def vector_rejection(a, b):
    return a - vector_projection(a, b)


def angle_between_v(v1, v2, direction=False):
    # Notice: only returns angles between 0 and 180 deg if direction == False

    try:
        v1 = v1.flatten()
        v2 = v2.flatten()

        n1 = v1 / np.linalg.norm(v1)
        n2 = v2 / np.linalg.norm(v2)
        ca = n1.dot(n2)

        if direction is not False:
            c = np.cross(n1, n2)
            d = c.dot(direction)
            ra = math.asin(np.clip(np.linalg.norm(c), -1, 1))

            if ca >= 0 and d >= 0:
                # 1st quadrant
                angle = ra
            elif ca <= 0 and d >= 0:
                # 2nd quadrant
                angle = np.pi - ra
            elif ca <= 0 and d <= 0:
                # 3rd quadrant
                angle = -np.pi + ra
            elif ca >= 0 and d <= 0:
                # 4th quadrant
                angle = -ra
            else:
                assert False, 'invalid logic: ca=%s, d=%s, ra=%s' % (ca, d, ra)
        else:
            angle = math.acos(np.clip(ca, -1, 1))
    except TypeError as e:
        raise Exception('Bad vectors:\n\tv1: %s\n\tv2: %s' % (v1, v2)) from e

    return angle


def angle_between_q(q1, q2):
    # from  https://chrischoy.github.io/research/measuring-rotation/
    qd = q1.conj() * q2
    return abs(wrap_rads(2 * math.acos(qd.normalized().w)))


def angle_between_rows(A, B, normalize=True):
    assert A.shape[1] == 3 and B.shape[1] == 3, 'matrices need to be of shape (n, 3) and (m, 3)'
    if A.shape[0] == B.shape[0]:
        # from https://stackoverflow.com/questions/50772176/calculate-the-angle-between-the-rows-of-two-matrices-in-numpy/50772253
        cos_angles = np.einsum('ij,ij->i', A, B)
        if normalize:
            p2 = np.einsum('ij,ij->i', A, A)
            p3 = np.einsum('ij,ij->i', B, B)
            cos_angles /= np.sqrt(p2 * p3)
    else:
        if normalize:
            A = A / np.linalg.norm(A, axis=1).reshape((-1, 1))
            B = B / np.linalg.norm(B, axis=1).reshape((-1, 1))
        cos_angles = B.dot(A.T)

    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def wrap_rads(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def if_none_q(w, x, y, z, fallback=None):
    c = list(map(lambda x: None if x == 'nan' else x, [w, x, y, z]))
    return fallback if np.any([v is None or np.isnan(v) for v in c]) else np.quaternion(*c)


def preprocess_image(data, gamma):
    data = np.atleast_3d(data)
    bot_v, top_v = np.quantile(data[:, :, 0], (0.0005, 0.9999))
    top_v = top_v * 1.2
    img = (data[:, :, 0] - bot_v) / (top_v - bot_v)
    if gamma != 1:
        img = np.clip(img, 0, 1) ** (1 / gamma)
    img = np.clip(255 * img + 0.5, 0, 255).astype(np.uint8)
    return img, (bot_v, top_v)


def rotate_array(arr, angle, new_size='same', border=cv2.BORDER_REPLICATE, border_val=None):
    arr = np.array(arr).squeeze()
    h, w, *c = arr.shape
    c = c[0] if len(c) > 0 else 1
    border_val = border_val if border_val is None else [border_val] * c

    if new_size == 'full':
        rh, rw = rot_arr_shape((h, w), angle)
    elif new_size == 'same':
        rw, rh = w, h
    else:
        rw, rh = new_size

    cx, cy = w / 2, h / 2
    mx = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), 1)
    mx[0, 2] += rw / 2 - cx
    mx[1, 2] += rh / 2 - cy
    rarr = cv2.warpAffine(arr, mx, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=border, borderValue=border_val)

    return rarr


def rot_arr_shape(shape, angle):
    h, w = shape
    rw = int((h * abs(math.sin(angle))) + (w * abs(math.cos(angle))))
    rh = int((h * abs(math.cos(angle))) + (w * abs(math.sin(angle))))
    return rh, rw


def rotate_aflow(aflow, old_shape2, angle1, angle2, new_size1='full', new_size2='full'):
    # rotate aflow content so that points to new rotated img1
    (oh2, ow2) = old_shape2

    if new_size2 == 'full':
        nh2, nw2 = rot_arr_shape((oh2, ow2), angle2)
    elif new_size2 == 'same':
        nw2, nh2 = ow2, oh2
    else:
        nw2, nh2 = new_size2

    R2 = np.array([[math.cos(angle2), -math.sin(angle2)],
                   [math.sin(angle2),  math.cos(angle2)]], dtype=np.float32)

    r_aflow = aflow - np.array([[[ow2 / 2, oh2 / 2]]], dtype=np.float32)
    r_aflow = r_aflow.reshape((-1, 2)).dot(R2).reshape(aflow.shape)
    r_aflow += np.array([[[nw2 / 2, nh2 / 2]]], dtype=np.float32)
    n_aflow = rotate_array(r_aflow, angle1, new_size=new_size1, border=cv2.BORDER_CONSTANT, border_val=np.nan)
    return n_aflow


def resize_aflow(aflow, size1, size2, sc2):
    w1, h1 = size1
    r_aflow = cv2.resize(aflow, size1, interpolation=cv2.INTER_NEAREST).reshape((-1, 2))

    w2, h2 = size2
    r_aflow = (r_aflow - np.array([[w2/sc2/2, h2/sc2/2]], dtype=r_aflow.dtype)) * sc2 \
               + np.array([[w2/2, h2/2]], dtype=r_aflow.dtype)

    # massage aflow
    r_aflow[np.any(r_aflow < 0, axis=1), :] = np.nan
    r_aflow[np.logical_or(r_aflow[:, 0] > w2 - 1, r_aflow[:, 1] > h2 - 1), :] = np.nan
    r_aflow = r_aflow.reshape((h1, w1, 2))

    return r_aflow


def rotate_expand_border(img, angle, fullsize=False, lib='opencv', to_pil=False):
    """
    opencv (v4.0.1) is fastest, pytorch (v1.8.1) on cpu is x20-25 slower, scipy (v1.6.0) is x28-36 slower
    """

    img = np.array(img).squeeze()
    h, w, *c = img.shape
    c = c[0] if len(c) > 0 else 1

    if fullsize:
        rw = int((h * abs(math.sin(angle))) + (w * abs(math.cos(angle))))
        rh = int((h * abs(math.cos(angle))) + (w * abs(math.sin(angle))))
    else:
        rw, rh = w, h

    if lib == 'opencv':
        cx, cy = w / 2, h / 2
        mx = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), 1)
        mx[0, 2] += rw / 2 - cx
        mx[1, 2] += rh / 2 - cy
        rimg = cv2.warpAffine(img, mx, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    elif lib == 'scipy':
        from scipy import ndimage
        rimg = ndimage.rotate(img, math.degrees(angle), reshape=fullsize, order=0, mode='nearest', prefilter=False)

    elif lib == 'pytorch':
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        img = TF.to_tensor(img)[None, ...]
        img.requires_grad = False

        theta = torch.tensor(angle)
        mx = torch.tensor([[[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]])

        grid = F.affine_grid(mx, [1, c, rh, rw], align_corners=False).type(img.dtype)
        rimg = F.grid_sample(img, grid, mode="nearest", padding_mode="border", align_corners=False)
        rimg = (rimg.permute([0, 2, 3, 1]).squeeze().numpy()*255 + 0.5).astype(np.uint8)
    else:
        assert False, 'invalid library selected for rotation: %s' % lib

    if to_pil:
        import PIL
        rimg = PIL.Image.fromarray(rimg)

    return rimg


class Camera:
    def __init__(self, matrix=None, resolution=None, pixel_size=None, focal_length=None,
                 center=None, f_num=None, dist_coefs=None):
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.center = center
        self.f_num = f_num
        self.matrix = matrix
        self.dist_coefs = np.array([0, 0, 0, 0] if dist_coefs is None else dist_coefs, dtype=float)
        self._inv_matrix = None

        if self.matrix is None:
            # camera borehole +z axis, up -y axis
            cx, cy = (resolution[0] / 2 - 0.5, resolution[1] / 2 - 0.5) if center is None else center
            fl_w, fl_h = focal_length if isinstance(focal_length, Iterable) else [focal_length] * 2
            fl_w, fl_h = fl_w/pixel_size, fl_h/pixel_size
            self.matrix = np.array([[fl_w, 0, cx],
                                    [0, fl_h, cy],
                                    [0, 0, 1]], dtype=float)

    @property
    def width(self):
        return self.resolution[0]

    @property
    def height(self):
        return self.resolution[1]

    @property
    def inv_matrix(self):
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self.matrix)
        return self._inv_matrix

    def project(self, pts3d):
        return self._project(pts3d, self.matrix, self.dist_coefs)

    @staticmethod
    def _project(P, K, dist_coefs):
        return cv2.projectPoints(P.reshape((-1, 3)), #np.hstack([P, np.ones((len(P), 1))]).reshape((-1, 1, 4)),
                                 np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 0], dtype=np.float32),
                                 K, np.array(dist_coefs),
                                 jacobian=False)[0].squeeze()

    def to_unit_sphere(self, ixy, undistort=True):
        return self.backproject(ixy, undistort=undistort)

    def backproject(self, ixy, dist=None, z_off=None, undistort=True):
        """ xi and yi are unaltered image coordinates, z_off is along the camera axis  """
        assert dist is None or z_off is None, "Use either dist or z_off. z_off is different from dist " \
                                              "in that it gives only the camera axis aligned component of the " \
                                              "vector going through a given pixel until intersecting the object."

        if undistort and self.dist_coefs is not None and np.sum(np.abs(self.dist_coefs)) > 0:
            ixy = self.undistort(ixy).squeeze()

        P = np.hstack((ixy + 0.5, np.ones((len(ixy), 1))))
        bP = self.inv_matrix.dot(P.T).T     # z-coordinates of points are all 1

        if z_off is not None:
            bP *= z_off.reshape((-1, 1))             # z-coordinates are at z_off
        else:
            bP = normalize_mx(bP)   # put points on the unit sphere

        if dist is not None:
            bP *= dist.reshape((-1, 1))              # extending point from unit sphere to be at distance `dist`

        return bP

    def undistort(self, P):
        return self._undistort(P, self.matrix, self.dist_coefs)

    @staticmethod
    def _undistort(P, cam_mx, dist_coefs):
        if len(P) > 0:
            pts = cv2.undistortPoints(P.reshape((-1, 1, 2)), cam_mx, np.array(dist_coefs), None, cam_mx)
            return pts
        return P


class ImageDB:
    def __init__(self, db_file: str, truncate: bool = False):
        self._conn = sqlite3.connect(db_file)
        self._cursor = self._conn.cursor()
        if truncate:
            self._cursor.execute("DROP TABLE IF EXISTS images")
            self._cursor.execute("""
                CREATE TABLE images (
                    id INTEGER PRIMARY KEY ASC NOT NULL,
                    rand REAL NOT NULL,
                    set_id INTEGER DEFAULT NULL,
                    file CHAR(128) NOT NULL,
                    sc_qw REAL DEFAULT NULL,    -- cam axis +x, up +z
                    sc_qx REAL DEFAULT NULL,
                    sc_qy REAL DEFAULT NULL,
                    sc_qz REAL DEFAULT NULL,
                    sc_sun_x REAL DEFAULT NULL, -- in icrf
                    sc_sun_y REAL DEFAULT NULL,
                    sc_sun_z REAL DEFAULT NULL,
                    trg_qw REAL DEFAULT NULL,   -- cam axis +x, up +z
                    trg_qx REAL DEFAULT NULL,
                    trg_qy REAL DEFAULT NULL,
                    trg_qz REAL DEFAULT NULL,
                    sc_trg_x REAL DEFAULT NULL, -- in icrf if sc_q given, else in cam frame
                    sc_trg_y REAL DEFAULT NULL,
                    sc_trg_z REAL DEFAULT NULL,
                    hz_fov REAL DEFAULT NULL,
                    img_angle REAL DEFAULT 0.0, -- rotation in rad around +x
                    vd REAL DEFAULT NULL,
                    cx1 REAL DEFAULT NULL,
                    cy1 REAL DEFAULT NULL,
                    cz1 REAL DEFAULT NULL,
                    cx2 REAL DEFAULT NULL,
                    cy2 REAL DEFAULT NULL,
                    cz2 REAL DEFAULT NULL,
                    cx3 REAL DEFAULT NULL,
                    cy3 REAL DEFAULT NULL,
                    cz3 REAL DEFAULT NULL,
                    cx4 REAL DEFAULT NULL,
                    cy4 REAL DEFAULT NULL,
                    cz4 REAL DEFAULT NULL
                )""")
            self._cursor.execute("DROP TABLE IF EXISTS subset")
            self._cursor.execute("""
                CREATE TABLE subset (
                    id INTEGER PRIMARY KEY ASC NOT NULL,
                    name CHAR(128) NOT NULL UNIQUE,
                    w INTEGER NOT NULL,
                    h INTEGER NOT NULL,
                    fx REAL NOT NULL,
                    fy REAL NOT NULL,
                    cx REAL NOT NULL,
                    cy REAL NOT NULL,
                    k1 REAL DEFAULT 0.0,
                    k2 REAL DEFAULT 0.0,
                    p1 REAL DEFAULT 0.0,
                    p2 REAL DEFAULT 0.0,
                    k3 REAL DEFAULT 0.0
            )""")
            self._conn.commit()
        else:
            r = self._cursor.execute("SELECT sql FROM sqlite_schema WHERE name = 'images'")
            sql = r.fetchone()[0]
            if 'sc_qw' not in sql:
                for col in ('sc_qw', 'sc_qx', 'sc_qy', 'sc_qz', 'sc_sun_x', 'sc_sun_y', 'sc_sun_z',
                            'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z'):
                    self._cursor.execute(f"ALTER TABLE images ADD COLUMN {col} REAL DEFAULT NULL")
                self._conn.commit()
            if 'hz_fov' not in sql:
                for col, default in (('hz_fov', 'NULL'), ('img_angle', '0')):
                    self._cursor.execute(f"ALTER TABLE images ADD COLUMN {col} REAL DEFAULT {default}")
                self._conn.commit()
            if 'set_id' not in sql:
                self._cursor.execute("ALTER TABLE images ADD COLUMN set_id INTEGER DEFAULT NULL")
                self._conn.commit()

    def add(self, fields: Tuple[str, ...], values: List[Tuple]) -> None:
        if len(values) == 0:
            return

        if 'rand' not in fields:
            fields = fields + ('rand',)
            for i in range(len(values)):
                values[i] = values[i] + (random.uniform(0, 1),)

        query = ("INSERT INTO images (" + ','.join(fields) + ") VALUES " +
                 ",".join([("(" + ",".join([('null' if v is None else ("'%s'" % v)) for v in row]) + ")")
                 for row in values]))
        self._cursor.execute(query)
        self._conn.commit()

    def delete(self, id):
        self._cursor.execute(f"DELETE FROM images WHERE id = {id}")
        self._conn.commit()

    def set(self, fields: Tuple[str, ...], values: List[Tuple], ignore: Tuple[str, ...] = None) -> None:
        if len(values) == 0:
            return

        assert 'id' in fields, '`id` field is required'
        ignore_on_update = ('id',) + (tuple() if ignore is None else ignore)
        if 'rand' not in fields:
            fields = fields + ('rand',)
            values = [(row + (random.uniform(0, 1),)) for row in values]
            ignore_on_update = ignore_on_update + ('rand',)

        self._cursor.execute(
            "INSERT INTO images (" + ','.join(fields) + ") VALUES " +
            ",".join([("(" + ",".join(['null' if v is None else ("'%s'" % str(v)) for v in row]) + ")")
                      for row in values]) +
            "ON CONFLICT(id) DO UPDATE SET " +
            ",\n".join(['%s = excluded.%s' % (f, f) for f in fields if f not in ignore_on_update])
        )
        self._conn.commit()

    def get(self, id: int, fields: Union[Tuple[str, ...], List[str]]) -> Tuple:
        r = self._cursor.execute(
            "SELECT " + ','.join(fields) + " FROM images WHERE id=%d" % id)
        row = r.fetchone()
        return row

    def get_all(self, fields: Union[Tuple[str, ...], List[str]],
                cond: str = None, start: float = 0, end: float = 1, ordered=True) -> List[Tuple]:
        query = ("SELECT " + ','.join(fields) +
                " FROM images WHERE rand >= %f AND rand < %f%s" % (start, end, '' if cond is None else ' AND %s' % cond) +
                (" ORDER BY rand" if ordered else ""))
        r = self._cursor.execute(query)
        rows = r.fetchall()
        return rows

    def query(self, fields: Union[Tuple[str, ...], List[str]], cond: str = None):
        query = "SELECT " + ','.join(fields) + " FROM images WHERE %s" % cond
        r = self._cursor.execute(query)
        row = r.fetchone()
        return row

    def set_subset(self, name, w, h, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        fields = ['name', 'w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']
        values = [[name, w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3]]
        ignore_on_update = ['name']

        self._cursor.execute(
            "INSERT INTO subset (" + ','.join(fields) + ") VALUES " +
            ",".join([("(" + ",".join(['null' if v is None else ("'%s'" % str(v)) for v in row]) + ")")
                      for row in values]) +
            "ON CONFLICT(id) DO UPDATE SET " +
            ",\n".join(['%s = excluded.%s' % (f, f) for f in fields if f not in ignore_on_update])
        )
        self._conn.commit()
        r = self._cursor.execute("SELECT id FROM subset WHERE name = '%s'" % name)
        row = r.fetchone()
        return int(row[0])

    def get_subset(self, id=None, name=None):
        assert (id or name) and (id and not name or not id and name), 'give either id or name'
        if id:
            cond = "id = %d" % int(id)
        else:
            cond = "name = '%s'" % name
        fields = ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']
        r = self._cursor.execute("SELECT " + ','.join(fields) + " FROM subset WHERE %s" % cond)
        row = r.fetchone()
        return row

    def __len__(self) -> int:
        r = self._cursor.execute("SELECT count(*) FROM images WHERE 1")
        count = r.fetchone()
        return count[0]


def nan_grid_interp(value_map, xy, max_radius=5.0, interp=False):
    """
    Interpolate values for sparse 2d-points based on the given value map
    """
    bad = ~(np.logical_and.reduce((xy[:, 1] < value_map.shape[0],
                                   xy[:, 0] < value_map.shape[1],
                                   xy[:, 1] >= 0, xy[:, 0] >= 0)))
    assert ~np.any(bad), 'out of bounds %s: %s' % (value_map.shape, (xy[bad, :]))

    H, W = value_map.shape
    uxy = unit_aflow(W, H).reshape((-1, 2))

    if interp:
        from scipy.interpolate import griddata
        value_map = value_map.flatten()
        values = griddata(uxy, value_map, xy, method='linear')
    else:
        ixy = (xy + 0.5).astype(int)
        values = value_map[ixy[:, 1], ixy[:, 0]]
        value_map = value_map.flatten()

    I = np.logical_not(np.isnan(value_map))
    I2 = np.isnan(values)
    interpolator = NearestKernelNDInterpolator(uxy[I, :], value_map[I], k_nearest=4 if interp else 1,
                                               max_distance=max_radius)
    missing = interpolator(xy[I2, :])
    values[I2] = missing

    return values


def estimate_pose_pnp(cam: Camera, trg_xyz, trg_ixy=None, ransac=False, max_err=1.0, ba=False,
                      ransac_method=cv2.SOLVEPNP_AP3P, input_in_opencv=False, repr_err_callback=None):
    if trg_ixy is None:
        assert len(trg_xyz.shape) == 3, "if feature observations not given, geometry need to be given as an HxWx3 array"
        h, w, _ = trg_xyz.shape
        trg_xyz = trg_xyz.reshape((-1, 3))
        trg_ixy = unit_aflow(w, h).reshape((-1, 2))
        I = np.where(np.logical_not(np.isnan(trg_xyz[:, 0])))[0]
        I = I[np.linspace(0, len(I), 100, endpoint=False).astype(int)]
        trg_xyz, trg_ixy = trg_xyz[I, :], trg_ixy[I, :]

    if not input_in_opencv:
        # convert from "axis +x, up +z" convention to "axis +z, up -y"
        trg_xyz = to_opencv_mx(trg_xyz)

    # opencv cam frame: axis +z, up -y
    if ba:
        import featstat.algo.model as fsm
        from featstat.algo.odo.simple import SimpleOdometry
        fs_cam = fsm.Camera(cam.width, cam.height, cam_mx=cam.matrix, dist_coefs=cam.dist_coefs)
        odo = SimpleOdometry(fs_cam, max_repr_err=max_err, repr_err=max_err, verbose=0)

        # TODO: optimize by adding a better suited method to SimpleOdometry
        pos, ori, kp_ids, kp_3d, succ_rate, track_len, repr_err, repr_err_ids = \
            odo.pair_stats_with_known_kp3d(np.arange(len(trg_xyz)), trg_ixy, trg_xyz)

        if ori is None:
            return [None] * 3

        Ie = repr_err_ids[:, 0] == 1
        inliers = repr_err_ids[Ie, 1][repr_err[Ie] < max_err]
        tvec, rvec = pos[-1], quaternion.as_rotation_vector(ori[-1])

        if repr_err_callback:
            repr_err_callback(trg_ixy[repr_err_ids[Ie, 1], :], repr_err[Ie])

    elif ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(trg_xyz, trg_ixy, cam.matrix, cam.dist_coefs,
                                                     iterationsCount=20000, reprojectionError=max_err,
                                                     flags=ransac_method)
        if not ok:
            return [None] * 3

        rvec, tvec = cv2.solvePnPRefineLM(trg_xyz[inliers, :], trg_ixy[inliers, :], cam.matrix,
                                          cam.dist_coefs, rvec, tvec)
    else:
        ok, rvec, tvec = cv2.solvePnP(trg_xyz, trg_ixy, cam.matrix, cam.dist_coefs)
        if not ok:
            return [None] * 3

    sc_trg_pos = tvec.flatten()
    sc_trg_ori = quaternion.from_rotation_vector(rvec.flatten())

    # convert to axis +x, up +z
    sc_trg_pos = from_opencv_v(sc_trg_pos)
    sc_trg_ori = from_opencv_q(sc_trg_ori)

    return (sc_trg_pos, sc_trg_ori) + ((inliers,) if ransac or ba else tuple())


def estimate_pose_icp(xyz0, xyz1, max_n1=None, max_n2=None):
    def limit_points(xyz0, xyz1, max_n):
        if len(xyz0) > max_n:
            I = np.floor(np.arange(0, len(xyz0) - 0.5, len(xyz0)/max_n)).astype(int)
            xyz0 = xyz0[I, :]
        if len(xyz1) > max_n:
            I = np.floor(np.arange(0, len(xyz1) - 0.5, len(xyz1)/max_n)).astype(int)
            xyz1 = xyz1[I, :]
        return xyz0, xyz1

    if max_n1:
        xyz0, xyz1 = limit_points(xyz0, xyz1, max_n1)

    ok, xyz0 = cv2.ppf_match_3d.computeNormalsPC3d(xyz0, 20, True, (0, 0, 0))
    ok, xyz1 = cv2.ppf_match_3d.computeNormalsPC3d(xyz1, 20, True, (0, 0, 0))

    if max_n2:
        xyz0, xyz1 = limit_points(xyz0, xyz1, max_n2)

    ok, err, T = cv2.ppf_match_3d_ICP().registerModelToScene(xyz1, xyz0)

    rel_pos = T[:3, 3]
    rel_ori = quaternion.from_rotation_matrix(T[:3, :3])
    return rel_pos, rel_ori, err


class NearestKernelNDInterpolator(NearestNDInterpolator):
    def __init__(self, x, y, k_nearest=None, kernel='gaussian', kernel_sc=None,
                 kernel_eps=1e-12, query_eps=0.05, max_distance=None, **kwargs):
        """
        Parameters
        ----------
        kernel : one of the following functions of distance that give weight to neighbours:
            'linear': (kernel_sc/(r + kernel_eps))
            'quadratic': (kernel_sc/(r + kernel_eps))**2
            'cubic': (kernel_sc/(r + kernel_eps))**3
            'gaussian': exp(-(r/kernel_sc)**2)
        k_nearest : uses k_nearest neighbours for interpolation
        """
        choices = ('linear', 'quadratic', 'cubic', 'gaussian')
        assert kernel in choices, 'kernel must be one of %s' % (choices,)
        self._tree_options = kwargs.get('tree_options', {})

        assert len(y.shape), 'only one dimensional `y` supported'
        assert not np.any(np.isnan(x)), 'does not support nan values in `x`'

        super(NearestKernelNDInterpolator, self).__init__(x, y, **kwargs)
        if kernel_sc is None:
            if k_nearest > 1:
                d, _ = self.tree.query(self.points, k=k_nearest)
                kernel_sc = np.mean(d) * k_nearest / (k_nearest - 1)
            else:
                assert max_distance is not None, 'kernel_sc or max_distance need to be set'
                kernel_sc = max_distance / 3

        if max_distance is None:
            max_distance = kernel_sc * 3

        self.kernel = kernel
        self.kernel_sc = kernel_sc
        self.kernel_eps = kernel_eps
        self.k_nearest = k_nearest
        self.max_distance = max_distance
        self.query_eps = query_eps

    def _linear(self, r):
        if scipy.sparse.issparse(r):
            return self.kernel_sc / (r + self.kernel_eps)
        else:
            return self.kernel_sc / (r + self.kernel_eps)

    def _quadratic(self, r):
        if scipy.sparse.issparse(r):
            return np.power(self.kernel_sc / (r.data + self.kernel_eps), 2, out=r.data)
        else:
            return (self.kernel_sc / (r + self.kernel_eps)) ** 2

    def _cubic(self, r):
        if scipy.sparse.issparse(r):
            return self.kernel_sc / (r + self.kernel_eps).power(3)
        else:
            return (self.kernel_sc / (r + self.kernel_eps)) ** 3

    def _gaussian(self, r):
        if scipy.sparse.issparse(r):
            return np.exp((-r.data / self.kernel_sc) ** 2, out=r.data)
        else:
            return np.exp(-(r / self.kernel_sc) ** 2)

    def __call__(self, *args):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        """
        from scipy.interpolate.interpnd import _ndim_coords_from_arrays

        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        r, idxs = self.tree.query(xi, self.k_nearest, eps=self.query_eps,
                                  distance_upper_bound=self.max_distance or np.inf)

        w = getattr(self, '_' + self.kernel)(r).reshape((-1, self.k_nearest)) + self.kernel_eps
        w /= np.sum(w, axis=1).reshape((-1, 1))

        # if idxs[i, j] == len(values), then i:th point doesnt have j:th match
        yt = np.concatenate((self.values, [np.nan]))

        yi = np.sum(yt[idxs] * w, axis=1)
        return yi
