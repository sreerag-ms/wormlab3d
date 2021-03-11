#!/usr/bin/env python

import sys
import numpy as np
import cv2

MAXFRAMES = -1
r = 2048


class VideoIterator:
    """
    wrapper for opencv video reader
    """

    def __init__(self, fn):
        # untested()

        self._vc = cv2.VideoCapture(fn)
        assert(self._vc.isOpened())

    def __iter__(self):
        return self

    def __next__(self):
        s, f = self._vc.read()
        if not s:
            # untested()
            raise StopIteration()

        return f, 0.


def open_file(fn):
    if fn[-4:] == ".seq":
        # TODO where is this file?
        # from .norpix import norpix
        seqfile = norpix.SeqFile(fn)
        image_data, timestamp = seqfile[0]
        return seqfile
    else:
        return VideoIterator(fn)


class Accumulate:
    """
    Implementation of low pass filter.
    """

    def __init__(self):
        self._run = np.ones([r, r], dtype=np.uint16)  # * 256*128
        self._output = np.ones([r, r], dtype=np.uint16)  # * 256*128

    def push(self, x):

        if len(x.shape) == 3:
            assert x[0, 0, 0] == x[0, 0, 1]
            assert x[4, 4, 0] == x[4, 4, 2]
            x = x[:, :, 0]

        x = x.astype(np.uint16)
        self._run += x*64 + (self._run >> 1)
        self._run >>= 1
        self._output = np.fmax(self._output, self._run)

    def get(self):
        a = self._output >> 7
        a = a.astype(np.uint8)
        return a


if __name__ == "__main__":
    seqfile = open_file(sys.argv[1])
    # trace("input", sys.argv[1])

    of = sys.argv[2]

    a = Accumulate()

    for k, (i, t) in enumerate(seqfile):
        if k == MAXFRAMES:
            break
        a.push(i)

    bg = a.get()
    print("done", k, bg.dtype, bg.max())

    cv2.imwrite(of, bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
