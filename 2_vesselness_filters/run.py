#!/usr/bin/env python3
"""
Usage:
  python detect_ends_vesselness.py \
    --input data/images/frame_51.jpg \
    --output_dir 1_convolution_method/outputs/vesselness
"""
import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from skimage.filters import frangi
from skimage.morphology import skeletonize


def compute_vesselness(img, scales=(1, 2, 4), beta1=0.5, beta2=15):
    """
    Apply Frangi vesselness filter at given scales.

    """
    # The frangi filter in skimage applies multi-scale filtering internally.
    vesselness = frangi(img, scale_range=(scales[0], scales[-1]), scale_step=scales[1]-scales[0], beta1=beta1, beta2=beta2)
    return vesselness


def threshold_vesselness(vesselness, thresh=None):
    """
    Threshold vesselness map to create a binary vessel mask.
    """
    if thresh is None:
        # use Otsu on normalized vesselness
        norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        _, thr = cv2.threshold((norm * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = thr / 255.0
    binary = vesselness >= thresh
    return binary


def find_skeleton_endpoints(mask):
    """
    Skeletonize a binary mask and return skeleton and endpoints.
    """
    # Skeletonize
    skel = skeletonize(mask)
    endpoints = []
    h, w = skel.shape
    coords = np.column_stack(np.where(skel))
    for y, x in coords:
        # extract 3x3 neighborhood
        y0, y1 = max(0, y-1), min(h, y+2)
        x0, x1 = max(0, x-1), min(w, x+2)
        nb = skel[y0:y1, x0:x1]
        if np.sum(nb) == 2:  # itself + one neighbor
            endpoints.append((x, y))
    return skel, endpoints


def save_image(img, path, normalize=False):
    """
    Save an image; optionally normalize to 0-255.
    """
    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main(args):
    # Create timestamped output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {args.input}")

    vesselness = compute_vesselness(img, scales=(1, 2, 4), beta1=0.5, beta2=15)
    save_image(vesselness, os.path.join(out_dir, 'vesselness.png'), normalize=True)

    mask = threshold_vesselness(vesselness, thresh=None)
    save_image((mask.astype(np.uint8)*255), os.path.join(out_dir, 'vessel_mask.png'))

    skel, endpoints = find_skeleton_endpoints(mask)
    save_image((skel.astype(np.uint8)*255), os.path.join(out_dir, 'skeleton.png'))

    annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for idx, (x, y) in enumerate(endpoints):
        color = (0, 0, 255) if idx == 0 else (255, 0, 0)
        cv2.circle(annotated, (x, y), radius=4, color=color, thickness=2)
    save_image(annotated, os.path.join(out_dir, 'annotated.png'))

    with open(os.path.join(out_dir, 'endpoints.txt'), 'w') as f:
        for (x, y) in endpoints:
            f.write(f"{x},{y}\n")

    print(f"Saved outputs to: {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/images/frame_4186.jpg')
    parser.add_argument('--output_dir', default='1_convolution_method/outputs/vesselness')
    args = parser.parse_args()
    main(args)
