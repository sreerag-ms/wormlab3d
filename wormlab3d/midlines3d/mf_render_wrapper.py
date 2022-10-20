from typing import Tuple, Optional, Union

import numpy as np
import torch

from wormlab3d.data.model import Reconstruction, Frame
from wormlab3d.data.model.mf_parameters import RENDER_MODE_GAUSSIANS
from wormlab3d.midlines3d.project_render_score import render_points, _taper_parameter
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import to_numpy


class RenderWrapper:
    """
    Helper class for plotting intermediate stages of the process.
    """
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
        self.scores_untapered: np.ndarray = None
        self.scores_tapered: np.ndarray = None
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

    def get_scores(self, tapered: bool = True):
        """
        Calculate the scores.
        """
        if self.scores_tapered is None:
            masks, blobs = self.get_masks_and_blobs()

            # Get targets
            images = self.frame.images
            images[images < self.reconstruction.mf_parameters.masks_threshold] = 0
            masks_target = images / images.max()

            # Normalise blobs
            sum_ = blobs.max(axis=(2, 3), keepdims=True)
            sum_ = sum_.clip(min=1e-8)
            blobs_normed = blobs / sum_

            # Score the points - look at projections in each view and check how well each blob matches against the lowest intensity image
            scores = (blobs_normed * masks_target[:, None]).sum(axis=(2, 3)).min(axis=0) \
                     / to_numpy(self.intensities[0]) \
                     / to_numpy(self.sigmas[0])  # Scale scores by sigmas and intensities
            scores_untapered = scores.copy()
            scores_tapered = to_numpy(_taper_parameter(torch.from_numpy(scores).unsqueeze(0))[0])
            self.scores_untapered = scores_untapered
            self.scores_tapered = scores_tapered

        if tapered:
            return self.scores_tapered
        else:
            return self.scores_untapered

    def get_detection_masks(
            self,
            threshold: float = 0.1,
            val_above_threshold: float = 1,
            val_below_threshold: float = 0.2,
    ) -> np.ndarray:
        """
        Get the detection masks.
        """
        masks, blobs = self.get_masks_and_blobs()
        scores = self.get_scores()

        # Normalise blobs
        sum_ = blobs.max(axis=(2, 3), keepdims=True)
        sum_ = sum_.clip(min=1e-8)
        blobs_normed = blobs / sum_

        # Make new render with blobs scaled by relative scores to get detection masks
        max_score = scores.max()
        rel_scores = np.where(max_score > 0, scores / max_score, np.ones_like(scores))
        sf = rel_scores[None, :, None, None]
        dm = (blobs_normed * sf).max(axis=1)
        dm[dm > threshold] = val_above_threshold
        dm[dm < threshold] = val_below_threshold

        return dm
