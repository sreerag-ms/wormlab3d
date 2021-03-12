
import numpy as np
# from contour import contour_mask_adaptive
import cv2
import torch
# from .trace import *
import torch.nn.functional as F


def smooth2d(c, k=7):
	sh = c.shape
	t = torch.ones([1,1,3,3])
	t[0,0,1,1] = k
	c = c.reshape([3,1,200,200])
	c = F.conv2d(c, t, padding=1)
	return c.reshape(sh)

def threshold(_m, radius=7, C=0, dilate=1, smooth=None, pow=1., cont_min=50,
              cont_adap=0):
	if pow!=1.:
		_m = _m.pow(pow)

	if smooth is not None:
		#untested()
		_m = smooth2d(_m, smooth)

	_m /= _m.max()
	m = _m.reshape([3,200,200]).numpy() * 255.
	m = m.astype(np.uint8)
	x = []
	for img in m:
		th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		                            cv2.THRESH_BINARY, radius, C)

		# needs to be bigger than cont_adap times the previous.
		cont = contour_mask_adaptive(th3, min_area=cont_min, adap=cont_adap)

		if dilate:
			cont = cv2.dilate(cont, None, iterations=dilate)

		x.append(torch.from_numpy(cont.astype(np.float32)))

	ts = torch.stack(x).reshape_as(_m)
	return ts/255.

def to_range(lox, hix, lo, hi):
	if(lox<0):
		hix -= lox
		lox = 0
	elif(hix>hi):
		lox -= hix - hi
		hix = hi

	return lox, hix

def maxBrightness(img):
	m = np.max(img)

#	img[i]=np.array(img[i])
	simg = img.astype(float)*(255./m)
	img = simg.astype(np.ubyte)
	return img

def my_imshow(x, annotate=None, wait=None):
	if annotate is not None:
		 x = cv2.circle(x.copy(), tuple(annotate), 7, (0,0,0), -1)

	try:
		y = cv2.resize(x, (1800,1800))
	except TypeError:
		print("type", type(x))
	
	cv2.imshow("frame0", y)

	if wait is None:
		cv2.waitKey()
	else:
		cv2.waitKey(wait)
