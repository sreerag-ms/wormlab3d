import os
from typing import Tuple, Optional, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Frame
from wormlab3d.data.model.mf_parameters import RENDER_MODE_GAUSSIANS
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.project_render_score import render_points, _taper_parameter
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import parse_target_arguments, to_numpy
from wormlab3d.trajectories.util import fetch_reconstruction

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'


def _get_targets() -> Tuple[Reconstruction, Frame]:
    """
    Resolve the reconstruction and frame.
    """
    args = parse_target_arguments()

    # Fetch reconstruction
    assert args.reconstruction is not None, 'Reconstruction must be specified!'
    reconstruction = fetch_reconstruction(reconstruction_id=args.reconstruction)
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions work for this!'

    # Fetch frame
    if args.frame_num is None:
        frame_num = np.random.randint(reconstruction.start_frame, reconstruction.end_frame)
        logger.info(f'Selected frame num = {frame_num} at random.')
    else:
        frame_num = args.frame_num
    frame = reconstruction.trial.get_frame(frame_num)

    return reconstruction, frame


def _get_reds(
        alpha_min: float = 0.3,
        white_at: float = 0.1
) -> np.ndarray:
    """
    Get the alpha-blended red colours.
    """
    cm = plt.get_cmap('Reds')
    reds = cm(np.linspace(0, 1, 256))
    reds[..., -1] = np.linspace(alpha_min, 1, 256)
    reds[:int(255 * white_at), :3] = (1, 1, 1)
    reds = (reds * 255).astype(np.uint8)
    return reds


def _make_blob_img(blob: np.ndarray, lut: np.ndarray, border_size: Optional[int],
                   border_colour: np.ndarray) -> Image.Image:
    """
    Make a blob image using colours from the look-up-table with a border as defined.
    """
    b = np.take(lut, blob, axis=0)
    if border_size > 0:
        b[:border_size, :] = border_colour
        b[-border_size:, :] = border_colour
        b[:, :border_size] = border_colour
        b[:, -border_size:] = border_colour
    b_img = Image.fromarray(b)
    return b_img


class RenderWrapper:
    sigma: torch.Tensor
    exponent: torch.Tensor
    intensity: torch.Tensor
    sigmas: torch.Tensor
    exponents: torch.Tensor
    intensities: torch.Tensor
    camera_sigmas: torch.Tensor
    camera_exponents: torch.Tensor
    camera_intensities: torch.Tensor

    def __init__(
            self,
            reconstruction: Reconstruction,
            frame: Frame,
    ):
        self.reconstruction = reconstruction
        self.frame = frame
        n = self.frame.frame_num
        self.ts = TrialState(self.reconstruction)
        self.points_2d = torch.from_numpy(self.ts.get('points_2d', n, n + 1).copy())
        self.init_params()

    def init_params(
            self,
            sigma: Optional[Union[float, Tuple[float, float]]] = None,
            exponent: Optional[float] = None,
            intensity: Optional[Union[float, Tuple[float, float]]] = None,
            camera_sigmas: Optional[float] = None,
            camera_exponents: Optional[float] = None,
            camera_intensities: Optional[float] = None,
    ):
        """
        Initialise the rendering parameters.
        """
        params = self.reconstruction.mf_parameters
        N = params.n_points_total
        N5 = int(N / 5)
        n = self.frame.frame_num

        if sigma is None:
            sigma = torch.from_numpy(self.ts.get('sigmas', n, n + 1)[0].copy())
            sigmas_min = params.sigmas_min
        else:
            if type(sigma) == tuple:
                sigmas_min = torch.tensor(sigma[0])
                sigma = torch.tensor(sigma[1])
            else:
                sigma = torch.tensor(sigma)
                sigmas_min = params.sigmas_min

        if exponent is None:
            exponent = torch.from_numpy(self.ts.get('exponents', n, n + 1).copy())
        else:
            exponent = torch.tensor(exponent)

        if intensity is None:
            intensity = torch.from_numpy(self.ts.get('intensities', n, n + 1).copy())
            intensities_min = params.intensities_min
        else:
            if type(intensity) == tuple:
                intensities_min = torch.tensor(intensity[0])
                intensity = torch.tensor(intensity[1])
            else:
                intensity = torch.tensor(intensity)
                intensities_min = params.intensities_min

        if camera_sigmas is None:
            self.camera_sigmas = torch.from_numpy(self.ts.get('camera_sigmas', n, n + 1).copy())
        else:
            self.camera_sigmas = torch.tensor(camera_sigmas)
        if camera_exponents is None:
            self.camera_exponents = torch.from_numpy(self.ts.get('camera_exponents', n, n + 1).copy())
        else:
            self.camera_exponents = torch.tensor(camera_exponents)
        if camera_intensities is None:
            self.camera_intensities = torch.from_numpy(self.ts.get('camera_intensities', n, n + 1).copy())
        else:
            self.camera_intensities = torch.tensor(camera_intensities)

        # Sigmas should be equal in the middle section but taper towards the ends
        sigma = sigma.clamp(min=sigmas_min)
        slopes = (sigma - sigmas_min) / N5 * torch.arange(N5)[None, :] + sigmas_min
        self.sigmas = torch.cat([
            slopes,
            torch.ones(1, N - 2 * N5) * sigma,
            slopes.flip(dims=(1,))
        ], dim=1)

        # Make exponents equal everywhere
        self.exponents = torch.ones(1, N) * exponent

        # Intensities should be equal in the middle section but taper towards the ends
        intensity = intensity.clamp(min=intensities_min)
        slopes = (intensity - intensities_min) / N5 \
                 * torch.arange(N5)[None, :] + intensities_min
        self.intensities = torch.cat([
            slopes,
            torch.ones(1, N - 2 * N5) * intensity,
            slopes.flip(dims=(1,))
        ], dim=1)

    def get_masks_and_blobs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Do rendering and get the blobs.
        """
        masks, blobs = render_points(
            self.points_2d.transpose(1, 2),
            self.sigmas,
            self.exponents,
            self.intensities,
            self.camera_sigmas,
            self.camera_exponents,
            self.camera_intensities,
            self.reconstruction.trial.crop_size,
            RENDER_MODE_GAUSSIANS,
        )

        return to_numpy(masks[0]), to_numpy(blobs[0])


def _get_masks_and_blobs(
        reconstruction: Reconstruction,
        frame: Frame,
        intensity_scaled_blobs: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the rendered masks and blobs for a midline.
    """
    renderer = RenderWrapper(reconstruction, frame)
    masks, blobs = renderer.get_masks_and_blobs()

    if intensity_scaled_blobs:
        intensities = renderer.intensities.repeat_interleave(3, 0)[:, :, None, None]
        sfs = renderer.camera_intensities.reshape(3)[:, None, None, None]
        intensities = intensities * sfs
        blobs *= intensities.numpy()

    return masks, blobs


def plot_blobs_individually(
        reconstruction: Reconstruction,
        frame: Frame,
        amplification_factor: float = 1.1,
        alpha_min: float = 0.3,
        white_at: float = 0.1,
        show_limit: int = 10,
):
    """
    Plot the rendered blobs for a midline.
    """
    trial = reconstruction.trial
    N = reconstruction.mf_parameters.n_points_total
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    masks, blobs = _get_masks_and_blobs(reconstruction, frame)

    # Amplify the blobs a little and convert to 8-bit
    blobs = ((blobs * amplification_factor).clip(max=1) * 255).astype(np.uint8)

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_blobs'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            amplification_factor=amplification_factor,
            alpha_min=alpha_min,
            white_at=white_at,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    shown = 0
    for c in range(3):
        if save_plots:
            os.makedirs(save_dir / f'c{c}', exist_ok=True)
        for n in range(N):
            blob = np.take(reds, blobs[c, n], axis=0)
            img = Image.fromarray(blob)
            if save_plots:
                img.save(save_dir / f'c{c}' / f'n={n:03d}.png')
            if show_plots and shown < show_limit:
                img.show()
                shown += 1


def plot_blob_stacks(
        reconstruction: Reconstruction,
        frame: Frame,
        amplification_factor: float = 1.1,
        alpha_min: float = 0.3,
        white_at: float = 0.1,
        border_size: int = 1,
        border_colour: Optional[Tuple[int]] = (0, 0, 0, 200),
        n_blobs: int = 33,
        blob_spacing: int = 6,
        blob_chunk_spacing: int = 40,
        show_first_n_blobs: int = 5,
        show_last_n_blobs: int = 1,
        n_dots: int = 3,
        dot_size: int = 2,
        dot_colour: Tuple[int] = (0, 0, 0, 200),
        highlight_idx: Optional[int] = None,
        highlight_offset: int = 100,
        highlight_border_size: int = 2,
):
    """
    Plot stacks of rendered blobs for a midline.
    """
    trial = reconstruction.trial
    N = reconstruction.mf_parameters.n_points_total
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    masks, blobs = _get_masks_and_blobs(reconstruction, frame)
    if border_colour is None:
        cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
        border_colours = cmap(np.linspace(0, 1, N)) * 255
    else:
        border_colours = np.ones((N, 4)) * border_colour

    # Amplify the blobs a little and convert to 8-bit
    blobs = ((blobs * amplification_factor).clip(max=1) * 255).astype(np.uint8)

    # Get the idxs of the blobs to use
    if 0 < n_blobs < N:
        blob_idxs = np.round(np.linspace(0, N - 1, n_blobs)).astype(int)
    else:
        blob_idxs = range(N)

    # Calculate the offsets
    offsets_first = np.arange(show_first_n_blobs) * blob_spacing
    offsets_last = np.arange(show_last_n_blobs) * blob_spacing
    highlight_rhs = trial.crop_size + highlight_offset \
                    + int((show_first_n_blobs - 0.5) * blob_spacing + blob_chunk_spacing / 2)

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_blob_stacks'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            amplification_factor=amplification_factor,
            alpha_min=alpha_min,
            white_at=white_at,
            border_size=border_size,
            border_colour=border_colour,
            n_blobs=n_blobs,
            blob_spacing=blob_spacing,
            blob_chunk_spacing=blob_chunk_spacing,
            show_first_n_blobs=show_first_n_blobs,
            show_last_n_blobs=show_last_n_blobs,
            n_dots=n_dots,
            dot_size=dot_size,
            dot_colour=dot_colour,
            highlight_idx=highlight_idx,
            highlight_offset=highlight_offset,
            highlight_border_size=highlight_border_size
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for c in range(3):
        dim_y = trial.crop_size + (show_first_n_blobs + show_last_n_blobs - 1) * blob_spacing + blob_chunk_spacing
        if highlight_idx is not None and dim_y < highlight_rhs:
            dim_x = highlight_rhs
        else:
            dim_x = dim_y
        bg = np.ones((dim_y, dim_x, 4), dtype=np.uint8) * 255
        blob_stack = Image.fromarray(bg)

        # Draw the back of the stack first
        for i, n in enumerate(blob_idxs[::-1][:show_last_n_blobs]):
            blob_img = _make_blob_img(blobs[c, n], reds, border_size, border_colours[n])
            blob_stack.paste(
                blob_img,
                box=(
                    blob_chunk_spacing + show_first_n_blobs * blob_spacing + offsets_last[show_last_n_blobs - i - 1],
                    offsets_last[i]
                ),
                mask=blob_img
            )

        # Add a highlighted frame if requested
        if highlight_idx is not None:
            n = blob_idxs[highlight_idx]
            blob_img = _make_blob_img(blobs[c, n], reds, highlight_border_size, border_colours[n])
            blob_stack.paste(
                blob_img,
                box=(
                    highlight_rhs - trial.crop_size,
                    highlight_rhs - trial.crop_size - highlight_offset
                ),
                mask=blob_img
            )

        # Draw the front of the stack
        for i, n in enumerate(blob_idxs[:show_first_n_blobs][::-1]):
            blob_img = _make_blob_img(blobs[c, n], reds, border_size, border_colours[n])
            blob_stack.paste(
                blob_img,
                box=(
                    offsets_first[show_first_n_blobs - i - 1],
                    blob_chunk_spacing + show_last_n_blobs * blob_spacing + offsets_first[i]
                ),
                mask=blob_img
            )

        # Add the dots
        dot_offsets = np.linspace(0, blob_chunk_spacing, n_dots + 2).round().astype(np.uint8)[1:-1]
        draw = ImageDraw.Draw(blob_stack)

        # Top-left dots
        for i in range(n_dots):
            x = blob_spacing * show_first_n_blobs + dot_offsets[i]
            y = blob_spacing * show_last_n_blobs + blob_chunk_spacing - dot_offsets[i]
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

        if highlight_idx is None:
            # Top-right dots
            for i in range(n_dots):
                x = trial.crop_size + blob_spacing * (show_first_n_blobs - 1) + dot_offsets[i]
                y = blob_spacing * show_last_n_blobs + blob_chunk_spacing - dot_offsets[i]
                draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

            # Bottom-right dots
            for i in range(n_dots):
                x = trial.crop_size + blob_spacing * (show_first_n_blobs - 1) + dot_offsets[i]
                y = trial.crop_size + blob_spacing * (show_last_n_blobs - 1) + blob_chunk_spacing - dot_offsets[i]
                draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

        if save_plots:
            blob_stack.save(save_dir / f'blob_stack_c{c}.png')
        if show_plots:
            blob_stack.show()


def plot_blob_stacks_horizontal(
        reconstruction: Reconstruction,
        frame: Frame,
        amplification_factor: float = 1.1,
        alpha_min: float = 0.3,
        white_at: float = 0.1,
        border_size: int = 1,
        border_colour: Optional[Tuple[int]] = (0, 0, 0, 200),
        n_blobs: int = 33,
        blob_spacing: int = 6,
        blob_chunk_spacing: int = 40,
        show_first_n_blobs: int = 5,
        show_last_n_blobs: int = 1,
        n_dots: int = 3,
        dot_size: int = 2,
        dot_colour: Tuple[int] = (0, 0, 0, 200),
):
    """
    Plot horizontal stacks of rendered blobs for a midline.
    """
    trial = reconstruction.trial
    N = reconstruction.mf_parameters.n_points_total
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    masks, blobs = _get_masks_and_blobs(reconstruction, frame)
    if border_colour is None:
        cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
        border_colours = cmap(np.linspace(0, 1, N)) * 255
    else:
        border_colours = np.ones((N, 4)) * border_colour

    # Amplify the blobs a little and convert to 8-bit
    blobs = ((blobs * amplification_factor).clip(max=1) * 255).astype(np.uint8)

    # Get the idxs of the blobs to use
    if 0 < n_blobs < N:
        blob_idxs = np.round(np.linspace(0, N - 1, n_blobs)).astype(int)
    else:
        blob_idxs = range(N)

    # Calculate the offsets
    offsets_first = np.arange(show_first_n_blobs) * blob_spacing
    offsets_last = trial.crop_size + blob_chunk_spacing \
                   + (show_first_n_blobs + np.arange(show_last_n_blobs)) * blob_spacing

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_blob_stacks_horizontal'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            amplification_factor=amplification_factor,
            alpha_min=alpha_min,
            white_at=white_at,
            border_size=border_size,
            border_colour=border_colour,
            n_blobs=n_blobs,
            blob_spacing=blob_spacing,
            blob_chunk_spacing=blob_chunk_spacing,
            show_first_n_blobs=show_first_n_blobs,
            show_last_n_blobs=show_last_n_blobs,
            n_dots=n_dots,
            dot_size=dot_size,
            dot_colour=dot_colour,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for c in range(3):
        dim_x = trial.crop_size * 2 + (show_first_n_blobs + show_last_n_blobs - 1) * blob_spacing + blob_chunk_spacing
        bg = np.ones((trial.crop_size, dim_x, 4), dtype=np.uint8) * 255
        bg[..., -1] = 0
        blob_stack = Image.fromarray(bg)

        # Draw the back of the stack first
        for i, n in enumerate(blob_idxs[::-1][:show_last_n_blobs]):
            blob_img = _make_blob_img(blobs[c, n], reds, border_size, border_colours[n])
            blob_stack.paste(
                blob_img,
                box=(offsets_last[show_last_n_blobs - i - 1], 0),
                mask=blob_img
            )

        # Draw the front of the stack
        for i, n in enumerate(blob_idxs[:show_first_n_blobs][::-1]):
            blob_img = _make_blob_img(blobs[c, n], reds, border_size, border_colours[n])
            blob_stack.paste(
                blob_img,
                box=(offsets_first[show_first_n_blobs - i - 1], 0),
                mask=blob_img
            )

        # Add the dots
        dot_offsets = np.linspace(0, blob_chunk_spacing + blob_spacing, n_dots + 2).round().astype(np.uint8)[1:-1]
        draw = ImageDraw.Draw(blob_stack)
        for i in range(n_dots):
            x = trial.crop_size + blob_spacing * (show_first_n_blobs - 1) + dot_offsets[i]
            y = trial.crop_size / 2
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

        if save_plots:
            blob_stack.save(save_dir / f'blob_stack_c{c}.png')
        if show_plots:
            blob_stack.show()


def plot_blob_intersections(
        reconstruction: Reconstruction,
        frame: Frame,
        blob_idx: int = 0,
        amplification_factor: float = 1.1,
        alpha_min: float = 0.3,
        white_at: float = 0.1
):
    """
    Plot the intersections between blobs for a vertex and an image.
    """
    trial = reconstruction.trial
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    masks, blobs = _get_masks_and_blobs(reconstruction, frame)
    blobs = blobs[:, blob_idx]

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_blob_{blob_idx}_intersections'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            amplification_factor=amplification_factor,
            alpha_min=alpha_min,
            white_at=white_at,
            blob_idx=blob_idx
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for c in range(3):
        blob = (blobs[c] / blobs[c].max() * amplification_factor).clip(max=1)
        img = frame.images[c]
        # img[img < reconstruction.mf_parameters.masks_threshold] = 0
        img /= img.max()

        # Calculate intersection and overlap "score" (not the real score, but close)
        intersect = blob * img
        score = intersect.sum()

        # Generate output
        img = ((1 - img) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        out = Image.fromarray(img)
        intersect = (intersect * 255).astype(np.uint8)
        intersect = np.take(reds, intersect, axis=0)
        mask = Image.fromarray(intersect)
        out.paste(mask, mask=mask)

        if save_plots:
            out.save(save_dir / f'c={c}_s={score:.2f}.png')
        if show_plots:
            out.show()


def plot_renders(
        reconstruction: Reconstruction,
        frame: Frame,
        alpha_min: float = 0.3,
        white_at: float = 0.1,
):
    """
    Plot the midline renders.
    """
    trial = reconstruction.trial
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    masks, blobs = _get_masks_and_blobs(reconstruction, frame)
    masks = (masks * 255).astype(np.uint8)

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_renders'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            alpha_min=alpha_min,
            white_at=white_at,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for c in range(3):
        mask = np.take(reds, masks[c], axis=0)
        img = Image.fromarray(mask)
        if save_plots:
            img.save(save_dir / f'c{c}.png')
        if show_plots:
            img.show()


def plot_render_variations(
        reconstruction: Reconstruction,
        frame: Frame,
        sigma_variants: Optional[List[Union[float, Tuple[float, float]]]] = None,
        intensity_variants: Optional[List[Union[float, Tuple[float, float]]]] = None,
        exponent_variants: Optional[List[float]] = None,
        alpha_min: float = 0.3,
        white_at: float = 0.1,
):
    """
    Plot variations on the midline renders.
    """
    trial = reconstruction.trial
    reds = _get_reds(alpha_min=alpha_min, white_at=white_at)
    renderer = RenderWrapper(reconstruction, frame)

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_render_variations'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            sigma_variants=sigma_variants,
            intensity_variants=intensity_variants,
            exponent_variants=exponent_variants,
            alpha_min=alpha_min,
            white_at=white_at,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    # Loop over parameters
    for param_name, variants in [
        ('sigma', sigma_variants),
        ('intensity', intensity_variants),
        ('exponent', exponent_variants)
    ]:
        # Reset parameters
        renderer.init_params()

        # Plot variants
        for variant in variants:
            renderer.init_params(**{param_name: variant})
            masks, _ = renderer.get_masks_and_blobs()
            masks = (masks * 255).astype(np.uint8)

            # Show/save renders
            for c in range(3):
                mask = np.take(reds, masks[c], axis=0)
                img = Image.fromarray(mask)
                if save_plots:
                    if type(variant) == tuple:
                        v_str = '[' + ','.join((f'{v:.2f}' for v in variant)) + ']'
                    else:
                        v_str = f'{variant:.2f}'
                    img.save(save_dir / f'{param_name}={v_str}_c{c}.png')
                if show_plots:
                    img.show()


def plot_rendering_parameters(
        reconstruction: Reconstruction,
        frame: Frame,
        sigma_variants: Optional[List[float]] = None,
        intensity_variants: Optional[List[float]] = None,
        exponent_variants: Optional[List[float]] = None,
):
    """
    Plot the rendering parameters.
    """
    trial = reconstruction.trial
    N = reconstruction.mf_parameters.n_points_total
    renderer = RenderWrapper(reconstruction, frame)

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_params'
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            sigma_variants=sigma_variants,
            intensity_variants=intensity_variants,
            exponent_variants=exponent_variants,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=1)

    fig, axes = plt.subplots(1, 3, figsize=(3.1, 0.95), gridspec_kw={
        'left': 0.1,
        'right': 0.98,
        'top': 0.95,
        'bottom': 0.2,
        'wspace': 0.38
    })

    for i in range(3):
        ax = axes[i]
        ax.spines['top'].set_visible(False)

        if i == 0:
            variants = sigma_variants
            lbl = '$\sigma$'
            param_name = 'sigma'
            state_name = 'sigmas'
        elif i == 1:
            variants = intensity_variants
            lbl = '$\iota$'
            param_name = 'intensity'
            state_name = 'intensities'
        else:
            variants = exponent_variants
            lbl = '$\\rho$'
            param_name = 'exponent'
            state_name = 'exponents'

        # Plot (tapered) parameter
        v = to_numpy(getattr(renderer, state_name)[0])
        ax.plot(v)

        if variants is not None:
            for variant in variants:
                r2 = RenderWrapper(reconstruction, frame)
                r2.init_params(**{param_name: variant})
                v2 = to_numpy(getattr(r2, state_name)[0])
                ax.plot(v2, linestyle=':')

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=0, right=N - 1)
        ax.set_xticks([0, N - 1])
        ax.set_xticklabels(['H', 'T'])

        # Set up y-axis
        # ax.set_ylim(bottom=0, top=v.max()*1.6)
        if v.max() == v.min():
            ax.set_yticks([0, v.max()])
            ax.set_yticklabels([None, f'{v.max():.2f}'])
        else:
            ax.set_yticks([0, v.min(), v.max()])
            ax.set_yticklabels([None, f'{v.min():.2f}', f'{v.max():.2f}'])

        # Add parameter label
        ax.text(-0.14, 0.96, lbl, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='top',
                fontweight='bold', fontsize=12)

    if save_plots:
        path = save_dir / f'params.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_scores(
        reconstruction: Reconstruction,
        frame: Frame,
        noise_scale: int = 0,
):
    """
    Plot the scores.
    """
    trial = reconstruction.trial
    N = reconstruction.mf_parameters.n_points_total
    renderer = RenderWrapper(reconstruction, frame)
    if noise_scale > 0:
        renderer.points_2d += torch.randn_like(renderer.points_2d) * noise_scale

    masks, blobs = renderer.get_masks_and_blobs()

    # Get targets
    images = frame.images
    images[images < reconstruction.mf_parameters.masks_threshold] = 0
    masks_target = images / images.max()

    # Normalise blobs
    sum_ = blobs.max(axis=(2, 3), keepdims=True)
    sum_ = sum_.clip(min=1e-8)
    blobs_normed = blobs / sum_

    # Score the points - look at projections in each view and check how well each blob matches against the lowest intensity image
    scores = (blobs_normed * masks_target[:, None]).sum(axis=(2, 3)).min(axis=0) \
             / to_numpy(renderer.intensities[0]) \
             / to_numpy(renderer.sigmas[0])  # Scale scores by sigmas and intensities
    scores_untapered = scores.copy()
    scores_tapered = to_numpy(_taper_parameter(torch.from_numpy(scores).unsqueeze(0))[0])

    # Prepare path
    save_dir = LOGS_PATH / (f'{START_TIMESTAMP}_scores'
                            + (f'_ns={noise_scale:.3E}' if noise_scale > 0 else '') +
                            f'_trial={trial.id}'
                            f'_frame={frame.frame_num}'
                            f'_reconstruction={reconstruction.id}')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=1, size=2)

    fig, axes = plt.subplots(2, figsize=(0.9, 1.6), gridspec_kw={
        'left': 0.23,
        'right': 0.94,
        'top': 0.98,
        'bottom': 0.11,
        'hspace': 0.38
    })

    for i in range(2):
        ax = axes[i]
        ax.spines['top'].set_visible(False)

        if i == 0:
            v = scores_untapered
            lbl = '$\mathcal{S}$'
        else:
            v = scores_tapered
            lbl = '$\hat{\mathcal{S}}$'

        for n in range(N - 1):
            ax.plot(ind[n:n + 2], v[n:n + 2], c=fc[n])

        ax.vlines(x=int(N / 2), ymin=0, ymax=v[int(N / 2)], linestyle=':', color='grey')

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=0, right=N - 1)
        ax.set_xticks([0, N - 1])
        ax.set_xticklabels(['H', 'T'])

        # Set up y-axis
        ax.set_ylim(bottom=0, top=scores_untapered.max() + 200)
        ax.set_yticks([0, 5000])

        # # Add parameter label
        # ax.text(-0.14, 0.96, lbl, transform=ax.transAxes,
        #         horizontalalignment='center', verticalalignment='top',
        #         fontweight='bold', fontsize=12)

    if save_plots:
        path = save_dir / f'scores.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    rec_, frame_ = _get_targets()

    # plot_blobs_individually(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     amplification_factor=1.1,
    #     alpha_min=0.3,
    #     white_at=0.1,
    #     show_limit=5,
    # )
    #
    # plot_blob_stacks(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     amplification_factor=1,
    #     alpha_min=0.7,
    #     white_at=0.05,
    #     border_size=3,
    #     # border_colour=(0, 0, 0, 200),
    #     border_colour=None,
    #     n_blobs=18,
    #     blob_spacing=20,
    #     blob_chunk_spacing=100,
    #     show_first_n_blobs=3,
    #     show_last_n_blobs=3,
    #     n_dots=3,
    #     dot_size=2,
    #     highlight_idx=9,
    #     highlight_offset=150,
    #     highlight_border_size=6,
    # )

    # plot_blob_stacks_horizontal(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     amplification_factor=1.,
    #     alpha_min=0.4,
    #     white_at=0.1,
    #     border_size=2,
    #     border_colour=None,
    #     n_blobs=16,
    #     blob_spacing=25,
    #     blob_chunk_spacing=150,
    #     show_first_n_blobs=7,
    #     show_last_n_blobs=3,
    #     n_dots=5,
    #     dot_size=3,
    # )

    # plot_blob_intersections(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     blob_idx=64,
    #     amplification_factor=1.1,
    #     alpha_min=0.5,
    #     white_at=0.02,
    # )

    # plot_renders(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     alpha_min=0.3,
    #     white_at=0.1,
    # )

    sigma_variants_ = [(0.005, 0.02), (0.05, 0.1)]
    intensity_variants_ = [(0.1, 0.3), (0.75, 1.3)]
    exponent_variants_ = [0.6, 3]

    # plot_render_variations(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     sigma_variants=sigma_variants_,
    #     intensity_variants=intensity_variants_,
    #     exponent_variants=exponent_variants_,
    #     alpha_min=0.7,
    #     white_at=0.02,
    # )

    # plot_rendering_parameters(
    #     reconstruction=rec_,
    #     frame=frame_,
    #     sigma_variants=sigma_variants_,
    #     intensity_variants=intensity_variants_,
    #     exponent_variants=exponent_variants_,
    # )

    # plot_rendering_parameters(
    #     reconstruction=rec_,
    #     frame=frame_,
    # )

    plot_scores(
        reconstruction=rec_,
        frame=frame_,
        noise_scale=0.5,
    )
