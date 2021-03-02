import os
import random
import sqlite3
from typing import Tuple, Union, List

import numpy as np
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


def load_aflow(fname):
    aflow = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.float32)
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

        self._cursor.execute(
            "INSERT INTO images (" + ','.join(fields) + ") VALUES " +
            ",".join([("('" + "','".join([str(v) for v in row]) + "')") for row in values])
        )
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
            ",".join([("('" + "','".join([str(v) for v in row]) + "')") for row in values]) +
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
                cond: str = None, start: float = 0, end: float = 1) -> List[Tuple]:
        r = self._cursor.execute(
            "SELECT " + ','.join(fields) +
            " FROM images WHERE rand >= %f AND rand < %f%s" % (start, end, '' if cond is None else ' AND %s' % cond))
        rows = r.fetchall()
        return rows

    def __len__(self) -> int:
        r = self._cursor.execute("SELECT count(*) FROM images WHERE 1")
        count = r.fetchone()
        return count[0]
