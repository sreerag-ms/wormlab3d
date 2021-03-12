#!/usr/bin/env mypython
# (couldn't make project work with python2)

# SGE_TASK_ID=11 ./sil_movies.sh 000000 bgu

from error import except_bad, eprint
import math
from .trace import incomplete, untested
from math import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
import numpy as np
from numpy.linalg import norm
from matplotlib.backend_bases import NavigationToolbar2, Event
import matplotlib.image as mpimg
import xml.etree.ElementTree
import sys
from sys import argv
import os
import getopt
import cv2

from .error import eprint
from .worm_parse import pvdparse
from .worm import worm

from .util import tail
from .pvdTD import total_displacement
from .pvdSD import self_dist_min
from .worm_util import worm_length
from .project import project32

# this is incomplete. use getopt below.
import parameters
import matplotlib.animation as manimation
from cv2 import VideoCapture

fps=25
frameSize = (2048, 2048)
cropSize = (200, 200)
max_frames = 1e99

class VideoIterator:
	def __iter__(self):
		return self
	def __init__(self, fn):
		untested()

		self._vc = VideoCapture(fn)
		if(not self._vc.isOpened()):
			print("not open >%s<" % fn)
		assert(self._vc.isOpened())

	def fps(self):
		return self._vc.get(cv2.CAP_PROP_FPS)
	def frameSize(self):
		incomplete()
		return (2048,2048)

#count = [vidcap[i].get(cv2.CAP_PROP_FRAME_COUNT) for i in range(3) ]

	def __next__(self):
		s, f = self._vc.read()
		if not s:
			untested()
			raise StopIteration()

		im = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
		return im

def open_file(fn):
	if fn[-4:] == ".seq":
		videoData = VideoData(fn)
#		seqfile = norpix.SeqFile(fn)
#		image_data, timestamp = seqfile[0]
#		print(type(image_data))
#		print(type(timestamp))
#		return seqfile
		return videoData
	else:
		try:
			return VideoIterator(fn)
		except IOError:
			print("something wrong", fn)



# TODO: use contour/find
def contour_mask(im, thresh=50, maxval=255, min_area=100):
	thresh=int(thresh)
#	print("contours", im.min(), im.max(), thresh)
	ret, _thresh = cv2.threshold(im, thresh, maxval, 0)
	#blocksize = 30
	#thresh = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 	#		            cv2.THRESH_BINARY,blocksize,2)

	im2, contours, hierarchy = cv2.findContours(_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#	contours = sorted(contours, key=cv2.contourArea, reverse=True)

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



parameters = parameters.parser(argv)

pvdDirname = parameters.pvd_dir + '/'
wpDirname = parameters.pvd_dir + '/'

from colormaps import phasecolor

skipnogait = True


# BUG: pass pvd as arg
# or tagfile?!
pvdFile = pvdDirname + '/skeletons_worm.pvd'

cam1=""
cam2=""
cam3=""
pcaFile=""
tagsfile=""

noplot=False

def usage():
	print("incomplete")

try:
	# the first few have been done by "parameter"
	opts, args = getopt.getopt(argv[1:], "aw:x:y:z:p:C:L:s:",
			 ["if=", "of=",
			 "bg=",
			 "mf=",
			 ])
	print("opts", opts, "args", args)
except getopt.GetoptError as err:
	#incomplete
	# print(help information and exit:
	print(str(err))  # will print something like "option -a not recognized"
	usage()
	sys.exit(2)

print("opts", opts)

calibrationfile = None
mode = "crop"
outdir = "."
stem = "nostem"

bgs=[None] * 3

for o, a in opts:
#	print("parse", o, a)
	if o == "-v":
		verbose = True
	elif o in ("-a"):
		skipnogait = False
	elif o in ("-s"):
		stem = a
	elif o in ("-n"):
		noplot = True
	elif o in ("-L"):
		mode = a
	elif o in ("-x"):
		cam1 = a
	elif o in ("-y"):
		cam2 = a
	elif o in ("-z"):
		cam3 = a
	elif o in ("--of"):
		ofile = a
	elif o in ("--mf"):
		max_frames = int(a)
	elif o in ("--if"):
		ifile = a
	elif o in ("--bg"):
		bgfn = a
	elif o in ("-C"):
		calibrationfile = a
	elif o in ("-p"):
		pcaFile=a
	elif o in ("--wpdir"):
		wpDirname=a
	elif o in ("--pvddir"):
		pvdDirname=a
	elif o in ("--outdir"):
		outdir = a
	elif o in ("--tagsfile"):
		tagsfile = a

from frame_tags import make_frame_tags
tags=make_frame_tags(tagsfile)

eprint("epvdDirname", pvdDirname)

# strip the last character
videoName = os.path.basename(pvdDirname)
eprint("videoName", pvdDirname)

# Set up axes and plot some awesome science

nCam = 3
camRange = range( 0, nCam )

def setupAxis( cam, ax, i ):
	if cam < 0:
		ax3d.set_xlabel( 'mm' )
		ax3d.set_ylabel( 'mm' )
		ax3d.set_zlabel( 'mm' )
		# ax.view_init(elev=20., azim=70 + i)
		# ax.grid(False)
	else:
		ax.set_title( str(cam) )
		ax.set_xlabel( 'Pixels' )
		ax.set_ylabel( 'Pixels' )
		# ax.set_title( "Camera View {0}".format(cam+1), fontsize=9 )


class VideoData:
	def __init__(self, filename):
		from .norpix import norpix
		self.seqfile = norpix.SeqFile(filename)

	def frameSize(self):
		incomplete()
		return (2048,2048)

	def fps(self):
		incomplete()
		return 25.
# fps = vidcap.get(cv2.CAP_PROP_FPS)

	def __iter__(self):
		class myIter:
			def __init__(self, i):
				self.i = iter(i)
			def __next__(self):
				data = next(self.i)
				print( data[1])
				return data[0]
			
		return myIter(self.seqfile)

	def getframe(self):
		incomplete()
		eprint("getting frame...")
		global captured_frames
		try:
			triplet=[]
			s=True
			for i in camRange:
				if not vidcap[i].isOpened():
					continue

				s,f = vidcap[i].read()
				if not s:
					if(i==0):
						eprint("something WRONG")
					eprint("done reading in frames")
					break
				sys.stdout.write('.')
				sys.stdout.flush()
	#			cv2.resize(f, (800, 600))
				triplet.append(f)
			if not s:
				return
			captured_frames = triplet
		except IOError:
			print("something went wrong")
			return
			pass



print("")

im0 = None
ims = [ None for _ in camRange ]

fs=25

metadata = dict(title=pvdFile, artist='leeds worm lab',
		comment=mode)

clipnumber = 0;
i=0
clipfile = outdir + "/" + stem + mode +"_sil.mp4"

from .img_util import to_range

# util?
def selcrop(center, framesize, cropSize):
	lox = int(center[0]) - cropSize[0]//2
	loy = int(center[1]) - cropSize[1]//2
	hix = int(center[0]) + cropSize[0]//2
	hiy = int(center[1]) + cropSize[1]//2

	lox, hix=to_range(lox,hix,0,framesize[0])
	loy, hiy=to_range(loy,hiy,0,framesize[1])
	return lox, hix, loy, hiy

def selembed(center, framesize, cropSize):
	lox = int(center[0]) - cropSize[0]//2
	loy = int(center[1]) - cropSize[1]//2
	hix = int(center[0]) + cropSize[0]//2
	hiy = int(center[1]) + cropSize[1]//2

	lox = max(0, lox)
	hix = min(hix, framesize[0])
	loy = max(0, loy)
	hiy = min(hiy, framesize[1])

	return lox, hix, loy, hiy

def embed(img, cog, cs, bgs):
	
	for i in range(3):
		bg=bgs[i].copy()
		lox, hix, loy, hiy = selembed(cog[i], bg.shape, cs)
		roi = img[i][loy:hiy, lox:hix]
		img[i] = bg
		img[i][loy:hiy, lox:hix] = roi

def crop(img, cog, cs):
	# print("cog", cog)
	for i in range(3):

		lox, hix, loy, hiy = selcrop(cog[i], (2048, 2048), cs)

		# print ("bounds", i, lox, hix, loy, hiy)
		#img[i] = img[i][lox:hix, loy:hiy]
		img[i] = img[i][loy:hiy, lox:hix]

skip_md = False

def extract_stuff(img, cropSize, backGround, prev_img=None):
	img = img.copy()
	from .img_util import maxBrightness
	img1 = cv2.subtract( backGround, img )
	maxb = img1.max()
	print("maxb", maxb, "/255")

	# contour of background-removed inverted image
	mask = contour_mask(img1, thresh=max(3,maxb*.05), maxval=maxb)

	if prev_img is None:
		pass
	elif skip_md:
		pass
	else:
		prev_img = prev_img.astype(np.longlong)
		#img = img.astype(np.longlong)
		delta = (prev_img - img.astype(np.longlong))**2
		delta = np.sqrt(delta)
		maxd = delta.max()
		print("max delta", maxd)
		mask2 = contour_mask(delta.astype(np.uint8), maxval=maxd, thresh=max(3,maxd*.5), min_area=10)

		# merge contour and motion masks
		mask = cv2.bitwise_or(mask, mask2)

	mask_dil = cv2.dilate(mask, None, iterations=10)

	print("final mask from", np.sum(mask_dil)/255, "pixels")
	mask = contour_mask(mask_dil, maxval=255, thresh=127, min_area=1000)
	print("has", np.sum(mask)/255, "pixels")

	mask = cv2.dilate(mask, None, iterations=10)

	demo=False

	if(demo):
		img = mask
	else:
		bg = cv2.bitwise_and(backGround, ~mask)
		img = cv2.bitwise_and(img, mask)
		img += bg

	return img


def do_it():
	import cv2
	videoData = open_file(ifile)

	##################
	fps = videoData.fps()
	outSize = videoData.frameSize()
	eprint("read", bgfn)
	backGround = cv2.imread(bgfn, cv2.IMREAD_GRAYSCALE)
	if(backGround is None):
		raise IOError("cannot open " + bgfn)
	assert(backGround is not None)
	eprint("read", bgfn, type(backGround))
	eprint("read", bgfn, backGround.shape)

	format="MJPG"
	fourcc = cv2.VideoWriter_fourcc(*format)
	fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
#	fourcc = cv2.VideoWriter_fourcc(*'X264') encoder not found
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
	fourcc = 0 # uncompressed/plain

	try:
		vidout = cv2.VideoWriter(ofile, apiPreference=0, fourcc=fourcc, fps=fps,
										  frameSize=outSize, isColor=False)
	except TypeError:
		eprint("somethings wrong with VideoWriter")
		vidout = cv2.VideoWriter(ofile, fourcc=fourcc, fps=fps,
										  frameSize=outSize, isColor=False)

	print("vidout", fps, outSize)
	assert(vidout.isOpened())

	img1 = None

	for i, img in enumerate(videoData):
		if i==max_frames:
			print("reached max frames", i)
			break

		iout = extract_stuff(img, cropSize, backGround, img1)
		img1 = img.copy()

#		eprint("write", img.shape, img.max(), type(img), img.dtype)
		assert(outSize==img.shape)
		vidout.write(iout)

	vidout.release()

if __name__=="__main__":
	do_it()
