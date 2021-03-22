import numpy as np


# todo: change class name and file name to something more suitable
# todo: add some comments to make this more understandable


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
        self._run += x * 64 + (self._run >> 1)
        self._run >>= 1
        self._output = np.fmax(self._output, self._run)

    def get(self):
        a = self._output >> 7
        a = a.astype(np.uint8)
        return a
