import math
import os
import random
import sqlite3
from typing import Tuple, Union, List, Iterable

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import cv2


def _find_files_recurse(root, path, samples, npy, ext, test, depth, relative):
    for fname in os.listdir(os.path.join(root, path)):
        fullpath = os.path.join(root, path, fname)
        if fname[-len(ext):] == ('.npy' if npy else ext):
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
    aflow = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.float32)
    aflow = aflow[:, :, :2]
    aflow[np.isclose(aflow, 2**16 - 1)] = np.nan
    return aflow / 8


def show_pair(img1, img2, aflow, file1='', file2='', pts=8):
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

    plt.show()


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

        if direction is not False:
            c = np.cross(n1, n2)
            d = c.dot(direction)
            s_angle = np.linalg.norm(c) * (-1 if d < 0 else 1)
        else:
            s_angle = n1.dot(n2)
    except TypeError as e:
        raise Exception('Bad vectors:\n\tv1: %s\n\tv2: %s' % (v1, v2)) from e

    return (math.asin if direction is not False else math.acos)(np.clip(s_angle, -1, 1))


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
        if self.matrix is None:
            # camera borehole +z axis, up -y axis
            cx, cy = (resolution[0] / 2, resolution[1] / 2) if center is None else center
            fl_w, fl_h = focal_length if isinstance(focal_length, Iterable) else [focal_length] * 2
            fl_w, fl_h = fl_w/pixel_size, fl_h/pixel_size
            self.matrix = np.array([[fl_w, 0, cx],
                                    [0, fl_h, cy],
                                    [0, 0, 1]], dtype=float)
        if self.dist_coefs is None:
            self.dist_coefs = np.array([0, 0, 0, 0])


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
            self._conn.commit()
        else:
            r = self._cursor.execute("SELECT sql FROM sqlite_schema WHERE name = 'images'")
            sql = r.fetchone()[0]
            if 'sc_qw' not in sql:
                for col in ('sc_qw', 'sc_qx', 'sc_qy', 'sc_qz', 'sc_sun_x', 'sc_sun_y', 'sc_sun_z',
                            'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z'):
                    self._cursor.execute(f"ALTER TABLE images ADD COLUMN {col} REAL DEFAULT NULL")
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
        if 'rand' not in fields:
            fields = fields + ('rand',)
            values = [(row + (random.uniform(0, 1),)) for row in values]

        self._cursor.execute(
            "INSERT INTO images (" + ','.join(fields) + ") VALUES " +
            ",".join([("(" + ",".join(['null' if v is None else ("'%s'" % str(v)) for v in row]) + ")")
                      for row in values]) +
            "ON CONFLICT(id) DO UPDATE SET " +
            ",\n".join(['%s = excluded.%s' % (f, f) for f in fields if f not in ('id', 'rand')])
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

    def __len__(self) -> int:
        r = self._cursor.execute("SELECT count(*) FROM images WHERE 1")
        count = r.fetchone()
        return count[0]
