import sys
import numpy as np
import cv2
from video_reader import open_file

MAXFRAMES = -1

class Accumulate:
    """
    Implementation of low pass filter.
    """

    def __init__(self, frame_size: tuple):
        self._run = np.ones(frame_size, dtype=np.uint16)
        self._output = np.ones(frame_size, dtype=np.uint16)

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
    frame_size = seqfile.frameSize

    of = sys.argv[2]

    a = Accumulate(frame_size)

    for k, image in enumerate(seqfile):
        if k == MAXFRAMES:
            break
        a.push(image)

    bg = a.get()
    print("done", k, bg.dtype, bg.max())

    cv2.imwrite(of, bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
