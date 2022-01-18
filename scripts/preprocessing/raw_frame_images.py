import os
import time

from PIL import Image
from wormlab3d import CAMERA_IDXS, LOGS_PATH
from wormlab3d.data.model.trial import Trial
from wormlab3d.toolkit.util import parse_target_arguments

show_imgs = False
save_imgs = True


def raw_frame_images():
    """
    Save raw camera frames.
    """
    args = parse_target_arguments()
    if args.trial is None:
        raise RuntimeError('This script must be run with the --trial=ID argument defined.')
    if args.frame_num is None:
        raise RuntimeError('This script must be run with the --frame-num=X argument defined.')

    # Fetch the trial and the video readers
    trial = Trial.objects.get(id=args.trial)
    frame = trial.get_frame(args.frame_num)
    reader = trial.get_video_triplet_reader()

    # Set the frame number and fetch the images from each video
    reader.set_frame_num(args.frame_num)
    images = reader.get_images()
    if len(images) != 3:
        raise RuntimeError('Raw image triplet not available.')

    # Show/save images
    for c in CAMERA_IDXS:
        img = Image.fromarray(images[c], 'L')

        if save_imgs:
            os.makedirs(LOGS_PATH, exist_ok=True)
            fn = time.strftime('%Y%m%d_%H%M') + f'_trial={trial.id}_frame={frame.frame_num}_c={c}.png'
            img.save(LOGS_PATH / fn)

        if show_imgs:
            img.show()


if __name__ == '__main__':
    raw_frame_images()
