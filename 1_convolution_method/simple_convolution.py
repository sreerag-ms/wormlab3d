#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from skimage.morphology import skeletonize


def create_conical_kernel(length=61, width=31, blur=5, flat_profile=False):
    kernel = np.ones((length, width), dtype=np.float32)
    width_center = width / 2.0
    for length_idx in range(length):
        normalized_position = length_idx / (length - 1)          
        current_half_width = (1 - normalized_position) * (width / 2)
        for width_idx in range(width):
            distance_from_center = abs(width_idx - width_center)
            if current_half_width > 0 and distance_from_center <= current_half_width:
                if flat_profile:
                    kernel[length_idx, width_idx] = 0.0
                else:
                    kernel[length_idx, width_idx] = distance_from_center / current_half_width
  
    kernel = cv2.GaussianBlur(kernel, (blur, blur), 0)
  
    kernel /= (kernel.sum() + 1e-8)
    return kernel


def preprocess_image(img_path, out_dir):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    img_med = cv2.medianBlur(img, 3)
    cv2.imwrite(os.path.join(out_dir, 'preprocessed.png'), img_med)

    _, bin_inv = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))  # Increased from 5x5 to 7x7
    mask = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_smooth)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel_smooth)

    mask_bool = mask.astype(bool)
    cv2.imwrite(os.path.join(out_dir, 'mask.png'), (mask_bool*255).astype(np.uint8))

    return img_med, mask_bool


def detect_kernel_tips(img_med, mask_bool, kernel, angles, suppress_r, out_dir):
    responses = []
    angle_responses = []
    
    # Generate responses for all angles
    for ang in angles:
        M = cv2.getRotationMatrix2D((kernel.shape[1]/2, kernel.shape[0]/2), ang, 1)
        k_rot = cv2.warpAffine(kernel, M, (kernel.shape[1], kernel.shape[0]), flags=cv2.INTER_LINEAR)
        resp = cv2.filter2D(img_med.astype(np.float32), -1, k_rot)
        responses.append(resp)
        angle_responses.append((ang, resp))

    stack = np.stack(responses, axis=-1)
    max_resp = np.max(stack, axis=-1)
    cv2.imwrite(os.path.join(out_dir, 'resp_max.png'), cv2.normalize(max_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    skel = skeletonize(mask_bool)
    cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), (skel.astype(np.uint8)*255))

    pts = np.column_stack(np.where(skel))
    endpoints = []
    
    h, w = skel.shape
    for y, x in pts:
        nb = skel[max(y-1,0):min(y+2,h), max(x-1,0):min(x+2,w)]
        if np.sum(nb) == 2:
            endpoints.append((x, y))

    # Determine tips and save only the max response angle for each tip
    if len(endpoints) >= 2:
        scores = [(max_resp[y, x], (x, y)) for x, y in endpoints]
        top = sorted(scores, key=lambda s: s[0], reverse=True)[:2]
        tips = [top[0][1], top[1][1]]
        
        # Save max response angle for each tip
        for i, tip in enumerate(tips):
            x, y = tip
            # Find angle with highest response at this tip position
            best_angle_idx = np.argmax([resp[y, x] for _, resp in angle_responses])
            best_angle, best_resp = angle_responses[best_angle_idx]
            
            # Save this specific angle response
            v = cv2.normalize(best_resp, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(out_dir, f"resp_tip{i+1}_angle{best_angle}.png"), v.astype(np.uint8))
    else:
        flat = max_resp.flatten()
        idx1 = np.argmax(flat)
        y1, x1 = np.unravel_index(idx1, max_resp.shape)
        mask2 = np.ones_like(max_resp, bool)
        y0 = max(0, y1 - suppress_r); y2 = min(max_resp.shape[0], y1 + suppress_r + 1)
        x0 = max(0, x1 - suppress_r); x2 = min(max_resp.shape[1], x1 + suppress_r + 1)
        mask2[y0:y2, x0:x2] = False
        tmp = max_resp.copy(); tmp[~mask2] = -np.inf
        y3, x3 = np.unravel_index(np.argmax(tmp), tmp.shape)
        tips = [(x1, y1), (x3, y3)]
        
        # Save max response angle for each tip
        for i, (x, y) in enumerate(tips):
            # Find angle with highest response at this tip position
            best_angle_idx = np.argmax([resp[y, x] for _, resp in angle_responses])
            best_angle, best_resp = angle_responses[best_angle_idx]
            
            # Save this specific angle response
            v = cv2.normalize(best_resp, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(out_dir, f"resp_tip{i+1}_angle{best_angle}.png"), v.astype(np.uint8))

    return tips, max_resp, skel


def extend_tips_along_skeleton(img_med, skel, tips, max_dist=30, bg_thresh=None):
    refined = []
    # Estimate background threshold if not provided
    if bg_thresh is None:
        bg_mean = np.mean(img_med[~skel])
        bg_thresh = bg_mean * 0.9

    for tip in tips:
        x, y = tip
        # find skeleton neighbor to get direction
        neigh = None
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                yy, xx = y+dy, x+dx
                if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1] and skel[yy, xx]:
                    neigh = (xx, yy)
                    break
            if neigh: break
        # compute unit vector
        if neigh:
            vec = np.array([x - neigh[0], y - neigh[1]], dtype=np.float32)
            norm = np.linalg.norm(vec)
            u = vec / (norm + 1e-8)
        else:
            u = np.array([0, 0], dtype=np.float32)

        last = (x, y)
        for i in range(1, max_dist+1):
            xi = int(round(x + u[0]*i))
            yi = int(round(y + u[1]*i))
            if xi < 0 or yi < 0 or yi >= img_med.shape[0] or xi >= img_med.shape[1]:
                break
            if img_med[yi, xi] > bg_thresh:
                break
            last = (xi, yi)
        refined.append(last)
    return refined


def visualize_kernel_at_endpoints(img_med, tips, kernel, angles, out_dir):
    """
    Generate images showing the conical kernel overlapped with endpoints at max response angles.
    """
    # Convert grayscale image to color for better visualization
    img_color = cv2.cvtColor(img_med, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y) in enumerate(tips):
        # Compute all angle responses to find the best one
        best_resp_value = float('-inf')
        best_angle = 0
        best_k_rot = None
        
        for ang in angles:
            # Rotate kernel to current angle
            M = cv2.getRotationMatrix2D((kernel.shape[1]/2, kernel.shape[0]/2), ang, 1)
            k_rot = cv2.warpAffine(kernel, M, (kernel.shape[1], kernel.shape[0]), flags=cv2.INTER_LINEAR)
            
            # Calculate response at this endpoint for this angle
            resp = cv2.filter2D(img_med.astype(np.float32), -1, k_rot)[y, x]
            
            if resp > best_resp_value:
                best_resp_value = resp
                best_angle = ang
                best_k_rot = k_rot.copy()
        
        # Create a colored version of the kernel for visualization
        k_vis = cv2.normalize(best_k_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        k_color = np.zeros((k_vis.shape[0], k_vis.shape[1], 3), dtype=np.uint8)
        k_color[:,:,2] = k_vis 
        
        kh, kw = kernel.shape
        x0 = int(x - kw/2)
        y0 = int(y - kh/2)
    
        endpoint_img = img_color.copy()
        
        y1, y2 = max(0, y0), min(img_color.shape[0], y0 + kh)
        x1, x2 = max(0, x0), min(img_color.shape[1], x0 + kw)
        ky1, ky2 = max(0, -y0), min(kh, img_color.shape[0] - y0)
        kx1, kx2 = max(0, -x0), min(kw, img_color.shape[1] - x0)
        
        overlay = endpoint_img[y1:y2, x1:x2].copy()
        kernel_region = k_color[ky1:ky2, kx1:kx2]
        mask = (kernel_region[:,:,2] > 20)  # Where kernel is non-trivial
        
        alpha = 0.6  # Kernel visibility (60%)
        overlay[mask] = overlay[mask] * (1-alpha) + kernel_region[mask] * alpha
        endpoint_img[y1:y2, x1:x2] = overlay
        
        cv2.circle(endpoint_img, (x, y), 3, (0, 0, 255), -1)
        
        cv2.putText(endpoint_img, f"Endpoint {i+1}, Angle: {best_angle}°", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(out_dir, f'endpoint_{i+1}_kernel_overlap.png'), endpoint_img)

def main(args):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_base, ts)
    os.makedirs(out_dir, exist_ok=True)

    img_med, mask_bool = preprocess_image(args.input, out_dir)

    # Build kernel and save
    kernel = create_conical_kernel(args.length, args.width, args.blur, flat_profile=True)
    cv2.imwrite(os.path.join(out_dir, 'kernel.png'), cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # Rough tip detection via kernel
    angles = list(range(0, 360, args.angle_step))
    tips, max_resp, skel = detect_kernel_tips(
        img_med, mask_bool, kernel,
        angles, args.suppress_r, out_dir
    )

    # Refine tips to capture faint gray ends
    refined = extend_tips_along_skeleton(
        img_med, skel, tips,
        max_dist=args.max_dist, bg_thresh=None
    )

    # Save final response map
    cv2.imwrite(os.path.join(out_dir, 'resp_max_final.png'),
                cv2.normalize(max_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # Annotate and save results
    orig = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    out_img = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    # rough tips
    cv2.circle(out_img, tips[0], 1, (0,0,255), 1)
    cv2.circle(out_img, tips[1], 1, (0,0,255), 1)
    # refined tips: gree
    for pt in refined:
        cv2.circle(out_img, pt, 1, (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(args.input)), out_img)

    print(f"Results saved to: {out_dir}")
    visualize_kernel_at_endpoints(img_med, tips, kernel, angles, out_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/images/frame_4186.jpg')
    parser.add_argument('--output_base', default='1_convolution_method/outputs/image')
    parser.add_argument('--length', type=int, default=41)
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--blur',   type=int, default=1,)
    parser.add_argument('--suppress_r', type=int, default=5)
    parser.add_argument('--angle_step', type=int, default=1)
    parser.add_argument('--max_dist',   type=int, default=30)
    args = parser.parse_args()
    main(args)