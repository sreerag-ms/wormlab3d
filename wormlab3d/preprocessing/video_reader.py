import numpy as np
import pims
import sys

class VideoIterator:
    """
    Generic video reader. This can read either any format supported by opencv or pims.
    """
    def __iter__(self):
        return self
    def __init__(self, fn: str):
        try:
            # standard video reader
            self.video = pims.PyAVReaderTimed(fn)
        except Exception as e:
            print(type(e), e)
            # TODO add specialist for seq files once we have test data
            # generic video reader
            self.video = pims.open(fn)

        self.current_frame : int = 0

    @property
    def fps(self):
        "Frames per second"
        return self.video.frame_rate

    @property
    def frameSize(self):
        shape = self.video.frame_shape
        if len(shape) == 3:
            shape = shape[:-1]
        return shape

    def __next__(self) -> pims.Frame:
        try:
            img = self.video[self.current_frame]
            grey = self._as_grey(img)
        except IndexError:
            raise StopIteration()
        self.current_frame += 1
        return grey

    @staticmethod
    @pims.pipeline
    def _as_grey(frame: pims.Frame) -> pims.Frame:
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        grey =  0.2125 * red + 0.7154 * green + 0.0721 * blue
        return grey.astype(frame.dtype)

def open_file(fn: str) -> VideoIterator:
    return VideoIterator(fn)

