import os

import r2d2
from r2d2.extract import load_network

from navex.models.base import BasePoint


class R2D2(BasePoint):
    DEFAULT_MODEL = os.path.join(os.path.dirname(r2d2.__file__), 'models', 'r2d2_WASF_N16.pt')

    def __init__(self, path=None):
        super(R2D2, self).__init__()
        self.super = load_network(path or R2D2.DEFAULT_MODEL)

    def forward(self, input):
        res = self.super([input])
        des = res['descriptors'][0]
        qlt = res['reliability'][0]
        det = res['repeatability'][0]
        return des, det, qlt
