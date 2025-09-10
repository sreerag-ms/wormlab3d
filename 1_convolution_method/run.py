#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import csv
import glob
from datetime import datetime
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

# ---------- kernels ----------
def create_tip_kernel(length=61, width=31, neg_cap=0.4, blur=3, flat_profile=True):
    h, w = length, width
    k = np.zeros((h, w), np.float32)

    cx = (w - 1) / 2.0

    for y in range(1, h):                    
        t = y / (h - 1)                      
        half = max(1.0, (1 - t) * (w / 2.0)) 
        for x in range(w):
            d = abs(x - cx)
            if d <= half:
                k[y, x] = 1.0 if flat_profile else (1 - d / half)

    k[0, :] = -neg_cap

    if blur and blur > 1:
        k = cv2.GaussianBlur(k, (blur | 1, blur | 1), 0)

    k -= k.mean()
    k /= (np.linalg.norm(k) + 1e-8)
    return k

def rotate_kernel_without_clipping(kernel, angle):
    h, w = kernel.shape
    
    diagonal = np.sqrt(h**2 + w**2)
    new_h, new_w = int(np.ceil(diagonal)), int(np.ceil(diagonal))
    
    dx = (new_w - w) / 2
    dy = (new_h - h) / 2
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    
    M[0, 2] += dx
    M[1, 2] += dy
    
    k_rot = cv2.warpAffine(kernel, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    
    return k_rot


def preprocess_image(img_path, out_dir=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    inv = img
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, 'inverted.png'), inv)

    inv = cv2.medianBlur(inv, 3)

    # CLAHE boosts faint tips without over-amplifying noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    inv_boost = clahe.apply(inv)
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, 'preprocessed.png'), inv_boost)

    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, ks)
    mask = cv2.erode(mask, ks)

    mask_bool = mask.astype(bool)
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, 'mask.png'), (mask_bool * 255).astype(np.uint8))
    return inv_boost, mask_bool


def skeleton_endpoints(mask_bool):
    skel = skeletonize(mask_bool).astype(np.uint8)
    nb_kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    nsum = convolve(skel, nb_kernel, mode='constant')
    ys, xs = np.where((skel == 1) & ((nsum == 11)))
    pts = list(zip(xs.tolist(), ys.tolist()))
    return pts[:2] if len(pts) >= 2 else pts, skel


# ---------- matching around seeds ----------
def tip_from_seed(img, kernel, seed_xy, angles, win=45):

    x0, y0 = seed_xy
    h, w = img.shape
    x1, x2 = max(0, x0 - win), min(w, x0 + win + 1)
    y1, y2 = max(0, y0 - win), min(h, y0 + win + 1)
    roi = img[y1:y2, x1:x2]
    best = (-np.inf, None, None, None)

    for ang in angles:
        k_rot = rotate_kernel_without_clipping(kernel, ang)
        
        resp = cv2.matchTemplate(roi.astype(np.float32), k_rot.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resp)
        if maxVal > best[0]:
            px, py = maxLoc
            
            center_x = x1 + px + k_rot.shape[1] // 2
            center_y = y1 + py + k_rot.shape[0] // 2
            
            rad_angle = np.deg2rad(ang)
            tip_offset = k_rot.shape[0] // 2
            tip_x = int(center_x + tip_offset * np.sin(rad_angle))
            tip_y = int(center_y + tip_offset * np.cos(rad_angle))

            best = (maxVal, tip_x, tip_y, ang)

    return best[1], best[2], best[3], best[0]


def detect_tips(img_med, mask_bool, kernel, angles, out_dir=None):
    # seeds from skeleton
    seeds, skel = skeleton_endpoints(mask_bool)
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), (skel * 255).astype(np.uint8))

    if len(seeds) < 2:
        angles_coarse = angles[::max(1, len(angles)//36)]
        h, w = img_med.shape
        resp_max = np.full((h, w), -np.inf, np.float32)
        for ang in angles_coarse:
            k_rot = rotate_kernel_without_clipping(kernel, ang)
            r = cv2.matchTemplate(img_med.astype(np.float32), k_rot.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            pad = ((k_rot.shape[0]//2, k_rot.shape[0]//2),
                   (k_rot.shape[1]//2, k_rot.shape[1]//2))
            r_pad = np.pad(r, pad, mode='edge')  # center back to image coords
            resp_max = np.maximum(resp_max, r_pad)
        idx1 = np.argmax(resp_max)
        y1, x1 = np.unravel_index(idx1, resp_max.shape)
        seeds = [(x1, y1)]
        # suppress and pick second
        suppress_r = 15
        mask2 = np.ones_like(resp_max, bool)
        mask2[max(0, y1 - suppress_r):min(resp_max.shape[0], y1 + suppress_r + 1),
              max(0, x1 - suppress_r):min(resp_max.shape[1], x1 + suppress_r + 1)] = False
        tmp = resp_max.copy()
        tmp[~mask2] = -np.inf
        idx2 = np.argmax(tmp)
        y2, x2 = np.unravel_index(idx2, tmp.shape)
        seeds.append((x2, y2))

    # refine each seed locally with full angle set
    tips = []
    for sx, sy in seeds[:2]:
        x, y, ang, score = tip_from_seed(img_med, kernel, (sx, sy), angles, win=45)
        tips.append((x, y, ang))
    
    # Return both tips and seeds for visualization
    return tips, seeds[:2]


def visualize_results(img_path, img_med, tips, kernel, out_dir, seeds=None, search_win=45):
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    out_img = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    # Visualize search windows if seeds are provided
    if seeds:
        search_img = out_img.copy()
        for i, (sx, sy) in enumerate(seeds):
            color = (0, 255, 255)
            # Draw the seed point
            cv2.circle(search_img, (sx, sy), 4, color, -1)
            
            # draw the search window around the seed
            x1 = max(0, sx - search_win)
            y1 = max(0, sy - search_win)
            x2 = min(orig.shape[1], sx + search_win)
            y2 = min(orig.shape[0], sy + search_win)
            cv2.rectangle(search_img, (x1, y1), (x2, y2), color, 2)
            
            # Label the seed
            cv2.putText(search_img, f"Seed {i+1}", (sx + 5, sy - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # save the visualization
        cv2.imwrite(os.path.join(out_dir, 'search_windows.png'), search_img)

    for i, (x, y, angle) in enumerate(tips):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(out_img, (x, y), 3, color, -1)

        L = 22
        end_x = int(x + L * np.cos(np.deg2rad(angle)))
        end_y = int(y + L * np.sin(np.deg2rad(angle)))
        cv2.line(out_img, (x, y), (end_x, end_y), color, 2)
        cv2.putText(out_img, f"Tip {i+1}", (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(os.path.join(out_dir, 'detected_tips.png'), out_img)

    for i, (x, y, angle) in enumerate(tips):
        k_rot = rotate_kernel_without_clipping(kernel, angle)
        k_vis = cv2.normalize(k_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img_color = cv2.cvtColor(img_med, cv2.COLOR_GRAY2BGR)
        overlay_img = img_color.copy()

        kh, kw = k_vis.shape
        
        tip_offset = kh // 2
        rad_angle = np.deg2rad(angle)
        x0 = int(x - tip_offset * np.sin(rad_angle) - kw/2)
        y0 = int(y - tip_offset * np.cos(rad_angle) - kh/2)
        
        y1, y2 = max(0, y0), min(img_color.shape[0], y0 + kh)
        x1, x2 = max(0, x0), min(img_color.shape[1], x0 + kw)
        ky1, ky2 = max(0, -y0), min(kh, img_color.shape[0] - y0)
        kx1, kx2 = max(0, -x0), min(kw, img_color.shape[1] - x0)

        k_color = np.zeros((kh, kw, 3), np.uint8)
        k_color[:, :, 2] = k_vis
        kernel_region = k_color[ky1:ky2, kx1:kx2]
        overlay = overlay_img[y1:y2, x1:x2].copy()
        mask = (kernel_region[:, :, 2] > 15)
        alpha = 0.6
        overlay[mask] = overlay[mask] * (1 - alpha) + kernel_region[mask] * alpha
        overlay_img[y1:y2, x1:x2] = overlay
        cv2.circle(overlay_img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(overlay_img, f"{int(round(angle))} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f'tip_{i+1}_kernel_overlay.png'), overlay_img)


def process_folder(input_folder, args):
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    
    kernel = create_tip_kernel(args.length, args.width, neg_cap=args.neg_cap, blur=args.blur, flat_profile=True)
    angles = list(range(0, 360, args.angle_step))
    
    csv_path = os.path.join(args.output_base, 'detection_results.csv')
    os.makedirs(args.output_base, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_position', 'frame_id', 'view', 'x_end1', 'y_end1', 'x_end2', 'y_end2'])
        
        for img_path in image_files:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            frame_position, view = filename.split('_')
            frame_position = int(frame_position)
            view = int(view)
            
            try:
                img_med, mask_bool = preprocess_image(img_path)
                tips, _ = detect_tips(img_med, mask_bool, kernel, angles)
                
                if len(tips) >= 2:
                    x1, y1, _ = tips[0]
                    x2, y2, _ = tips[1]
                else:
                    x1 = y1 = x2 = y2 = -1
                
                writer.writerow([frame_position, frame_position, view, x1, y1, x2, y2])
                print(f"Processed {filename}: tips at ({x1},{y1}) and ({x2},{y2})")
                
            except Exception as e:
                writer.writerow([frame_position, frame_position, view, -1, -1, -1, -1])
                print(f"Error processing {filename}: {e}")
    
    print(f"Results saved to: {csv_path}")

def main(args):
    if args.folder:
        process_folder(args.folder, args)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(args.output_base, ts)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing {args.input}...")
        img_med, mask_bool = preprocess_image(args.input, out_dir)

        kernel = create_tip_kernel(args.length, args.width, neg_cap=args.neg_cap, blur=args.blur, flat_profile=True)
        cv2.imwrite(os.path.join(out_dir, 'kernel.png'),
                    cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        angles = list(range(0, 360, args.angle_step))
        tips, seeds = detect_tips(img_med, mask_bool, kernel, angles, out_dir)

        visualize_results(args.input, img_med, tips, kernel, out_dir, seeds=seeds, search_win=45)
        print(f"Results saved to: {out_dir}")
        print(f"Detected tips at: {tips}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Directional template matching for worm tip detection')
    parser.add_argument('--input', default='output/images/frames_001000_001600/001547_0.png')
    parser.add_argument('--folder', help='Process all PNG images in this folder')
    parser.add_argument('--output_base', default='1_convolution_method/outputs/kernel_only')
    parser.add_argument('--length', type=int, default=26)
    parser.add_argument('--width', type=int, default=18)
    parser.add_argument('--blur', type=int, default=3)
    parser.add_argument('--neg_cap', type=float, default=0.4, help='strength of negative cap ahead of tip (0..1)')
    parser.add_argument('--angle_step', type=int, default=1)
    args = parser.parse_args()
    main(args)
