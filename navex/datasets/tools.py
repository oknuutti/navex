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
            else:
                print('rejected: %s' % fullpath)
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


def ypr_to_q(dec, ra, cna):
    if dec is None or ra is None or cna is None:
        return None

    # intrinsic euler rotations z-y'-x'', first right ascencion, then declination, and last celestial north angle
    return (
            np.quaternion(math.cos(ra / 2), 0, 0, math.sin(ra / 2))
            * np.quaternion(math.cos(-dec / 2), 0, math.sin(-dec / 2), 0)
            * np.quaternion(math.cos(-cna / 2), math.sin(-cna / 2), 0, 0)
    )


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
    return aqqmx[:, 1:]


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


def nadir_unit_v(sc_trg_q):
    # following assumed, not certain if necessary though:
    #   - cam: +x bore, +z up
    #   - trg: +x zero lat & lon, +z north pole
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


def wrap_rads(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def preprocess_image(data, gamma):
    data = np.atleast_3d(data)
    bot_v, top_v = np.quantile(data[:, :, 0], (0.0005, 0.9999))
    top_v = top_v * 1.2
    img = (data[:, :, 0] - bot_v) / (top_v - bot_v)
    if gamma != 1:
        img = np.clip(img, 0, 1) ** (1 / gamma)
    img = np.clip(255 * img + 0.5, 0, 255).astype(np.uint8)
    return img, (bot_v, top_v)


def rotate_array(arr, angle, fullsize=False, border=cv2.BORDER_REPLICATE, border_val=None):
    arr = np.array(arr).squeeze()
    h, w, *c = arr.shape
    c = c[0] if len(c) > 0 else 1
    border_val = border_val if border_val is None else [border_val] * c

    if fullsize:
        rh, rw = rot_arr_shape((h, w), angle)
    else:
        rw, rh = w, h

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


def rotate_aflow(aflow, shape2, angle1, angle2, legacy=False):
    # rotate aflow content so that points to new rotated img1
    (oh2, ow2), (nh2, nw2) = shape2, rot_arr_shape(shape2, angle2)
    R2 = np.array([[math.cos(angle2), -math.sin(angle2)],
                   [math.sin(angle2),  math.cos(angle2)]], dtype=np.float32)

    r_aflow = aflow - np.array([[[ow2 / 2, oh2 / 2]]], dtype=np.float32)
    r_aflow = r_aflow.reshape((-1, 2)).dot(R2.T.T).reshape(aflow.shape)     # not sure why need extra .T
    r_aflow = r_aflow + np.array([[[nw2 / 2, nh2 / 2]]], dtype=np.float32)

    if legacy:
        # TODO: remove legacy code when new way validated
        # rotate aflow indices same way as img0 was rotated
        import scipy.interpolate as interp
        R1 = np.array([[math.cos(angle1), -math.sin(angle1)],
                       [math.sin(angle1), math.cos(angle1)]], dtype=np.float32)
        (oh1, ow1), (nh1, nw1) = aflow.shape[:2], rot_arr_shape(aflow.shape[:2], angle1)

        ifun = interp.RegularGridInterpolator((np.arange(-oh1 / 2, oh1 / 2, dtype=np.float32),
                                               np.arange(-ow1 / 2, ow1 / 2, dtype=np.float32)), r_aflow,
                                              method="nearest", bounds_error=False, fill_value=np.nan)

        grid = unit_aflow(nw1, nh1) - np.array([[[nw1 / 2, nh1 / 2]]])
        grid = grid.reshape((-1, 2)).dot(np.linalg.inv(R1).T).reshape((nh1, nw1, 2))
        n_aflow = ifun(np.flip(grid, axis=2).astype(np.float32))
    else:
        n_aflow = rotate_array(r_aflow, angle1, fullsize=True, border=cv2.BORDER_REPLICATE, border_val=np.nan)

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
        self.dist_coefs = dist_coefs
        self._inv_matrix = None

        if self.matrix is None:
            # camera borehole +z axis, up -y axis
            cx, cy = (resolution[0] / 2 - 0.5, resolution[1] / 2 - 0.5) if center is None else center
            fl_w, fl_h = focal_length if isinstance(focal_length, Iterable) else [focal_length] * 2
            fl_w, fl_h = fl_w/pixel_size, fl_h/pixel_size
            self.matrix = np.array([[fl_w, 0, cx],
                                    [0, fl_h, cy],
                                    [0, 0, 1]], dtype=float)
        if self.dist_coefs is None:
            self.dist_coefs = np.array([0, 0, 0, 0])

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

    def backproject(self, xi, yi, z_off, undistort=True):
        """ xi and yi are unaltered image coordinates, z_off is usually negative  """
        single = isinstance(xi, (int, float))

        if undistort and self.dist_coefs is not None and np.sum(np.abs(self.dist_coefs)) > 0:
            P = np.array([[xi, yi]]) if single else np.hstack((xi[:, None], yi[:, None]))
            uP = self.undistort(P)
            xi, yi = uP[0, 0, :] if single else uP.squeeze().T

        xi, yi = xi + 0.5, yi + 0.5
        P = np.array([[xi, yi, 1]]) if single else np.hstack((xi[:, None], yi[:, None], np.ones((len(xi), 1))))
        bP = self.inv_matrix.dot(P.T) * z_off
        return tuple(bP.flatten()) if single else bP.T

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
                    sc_qw REAL DEFAULT NULL,
                    sc_qx REAL DEFAULT NULL,
                    sc_qy REAL DEFAULT NULL,
                    sc_qz REAL DEFAULT NULL,
                    sc_sun_x REAL DEFAULT NULL,
                    sc_sun_y REAL DEFAULT NULL,
                    sc_sun_z REAL DEFAULT NULL,
                    trg_qw REAL DEFAULT NULL,
                    trg_qx REAL DEFAULT NULL,
                    trg_qy REAL DEFAULT NULL,
                    trg_qz REAL DEFAULT NULL,
                    sc_trg_x REAL DEFAULT NULL,
                    sc_trg_y REAL DEFAULT NULL,
                    sc_trg_z REAL DEFAULT NULL,
                    hz_fov REAL DEFAULT NULL,
                    img_angle REAL DEFAULT 0.0,
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

    def set(self, fields: Tuple[str, ...], values: List[Tuple]) -> None:
        if len(values) == 0:
            return

        assert 'id' in fields, '`id` field is required'
        ignore_on_update = ('id',)
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
