import cv2
import matplotlib.pyplot as plt
import numpy as np

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT


def generate_annotated_images(
        image_triplet: np.ndarray,
        points_2d: np.ndarray
) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = []
    cmap_midline = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = np.array([cmap_midline(i) for i in np.linspace(0, 1, points_2d.shape[0])])
    colours = np.round(colours * 255).astype(np.uint8)
    for i, img_array in enumerate(image_triplet):
        z = ((1 - img_array) * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)

        # Overlay 2d projection
        p2d = points_2d[:, i]
        for j, p in enumerate(p2d):
            z = cv2.drawMarker(
                z,
                p,
                color=colours[j].tolist(),
                markerType=cv2.MARKER_CROSS,
                markerSize=3,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(
                    z,
                    p2d[j - 1],
                    p2d[j],
                    color=colours[j].tolist(),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        images.append(z)
    images = np.fliplr(np.concatenate(images))

    return images.transpose(1, 0, 2)
