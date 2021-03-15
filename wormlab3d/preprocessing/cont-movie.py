#!/usr/bin/env python
import cv2
import getopt
import numpy as np
import sys
from typing import Optional

def usage():
        #TODO
        print(f'{sys.argv[0]}' + ' --if={input.raw} --bg={input.background} --of={output.video_1}')

class VideoIterator:
    def __iter__(self):
        return self
    def __init__(self, fn):
        # untested()

        self._vc = cv2.VideoCapture(fn)
        if(not self._vc.isOpened()):
            print("not open >%s<" % fn)
        assert(self._vc.isOpened())

    def fps(self):
        return self._vc.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def frameSize():
        # incomplete()
        return (2048,2048)

    def __next__(self):
        s, f = self._vc.read()
        if not s:
            # untested()
            raise StopIteration()

        im = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return im

def open_file(fn: str) -> VideoIterator:
    if fn[-4:] == ".seq":
        videoData = VideoData(fn)
#        seqfile = norpix.SeqFile(fn)
#        image_data, timestamp = seqfile[0]
#        print(type(image_data))
#        print(type(timestamp))
#        return seqfile
        return videoData
    else:
                return VideoIterator(fn)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def contour_mask(im: np.array,
                 thresh: int = 50,
                 maxval: int = 255,
                 min_area: int = 100) -> np.array:
    """
    Create mask from threshold contour.
    """
    # find contours by threshold
    thresh=int(thresh)
    _, _thresh = cv2.threshold(im, thresh, maxval, 0)
    contours, _ = cv2.findContours(_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # construct mask
    mask = np.zeros_like(im)

    pc = nc = 0
    for c in contours:
            nc+=1
            area = cv2.contourArea(c)
            if area < min_area:
                    pass
            else:
                    cv2.drawContours(mask, [c], 0, 255, -1)
                    pc += 1

    print(pc, "out of", nc, "contours, thresh=", thresh)

    return mask


def extract_stuff(img: np.array,
                  backGround: np.array,
                  prev_img: Optional[np.array] = None):
    """
    Create output image of backGround plus masked input image.
    The mask is determined as a dilated combination of background subtraction and motion filters.
    """

    img = img.copy()
    img1 = cv2.subtract( backGround, img )
    maxb = img1.max()

    # contour of background-removed inverted image
    mask = contour_mask(img1, thresh=max(3,maxb*.05), maxval=maxb)

    # create motion mask
    if prev_img is None:
            pass
    else:
            prev_img = prev_img.astype(np.longlong)
            delta = (prev_img - img.astype(np.longlong))**2
            delta = np.sqrt(delta)
            maxd = delta.max()
            mask2 = contour_mask(delta.astype(np.uint8), maxval=maxd, thresh=max(3,maxd*.5), min_area=10)

            # merge contour and motion masks
            mask = cv2.bitwise_or(mask, mask2)

    # dilate mask
    mask_dil = cv2.dilate(mask, None, iterations=10)

    # contour mask again
    print("final mask from", np.sum(mask_dil)/255, "pixels")
    mask = contour_mask(mask_dil, maxval=255, thresh=127, min_area=1000)
    print("has", np.sum(mask)/255, "pixels")

    # dilate mask
    mask = cv2.dilate(mask, None, iterations=10)

    # apply output image is background + masked image
    bg = cv2.bitwise_and(backGround, ~mask)
    img = cv2.bitwise_and(img, mask)
    img += bg

    return img

def do_it(ifile: str, bgfn: str, ofile: str):
    # load video
    videoData = open_file(ifile)
    fps = videoData.fps()
    outSize = videoData.frameSize()

    # load background
    backGround = cv2.imread(bgfn, cv2.IMREAD_GRAYSCALE)
    if(backGround is None):
        raise IOError("cannot open " + bgfn)
    assert(backGround is not None)

    # open output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        vidout = cv2.VideoWriter(ofile, apiPreference=0, fourcc=fourcc, fps=fps,
                                          frameSize=outSize, isColor=False)
    except TypeError:
        print("somethings wrong with VideoWriter")
        vidout = cv2.VideoWriter(ofile, fourcc=fourcc, fps=fps,
                                          frameSize=outSize, isColor=False)
    assert(vidout.isOpened())

    # create output video frames and save
    img1 = None
    for img in videoData:
        iout = extract_stuff(img, backGround, img1)
        img1 = img.copy()

        assert(outSize==img.shape)
        vidout.write(iout)

    vidout.release()
    print(f'{ofile} complete')

if __name__=="__main__":
    # process arguments
    try:
            opts, args = getopt.getopt(sys.argv[1:], "",
                                       ["if=", "of=",
                                        "bg=",
                                        ])
            print("opts", opts, "args", args)
    except getopt.getopterror as err:
            #incomplete
            # print(help information and exit:
            eprint(str(err))  # will print something like "option -a not recognized"
            usage()
            sys.exit(2)

    ifile = ""
    ofile = ""
    bgfile = ""

    for o, a in opts:
        if o == "-v":
                verbose = True
        elif o in ("--of"):
                ofile = a
        elif o in ("--if"):
                ifile = a
        elif o in ("--bg"):
                bgfile = a

    do_it(ifile=ifile, ofile=ofile, bgfn=bgfile)
