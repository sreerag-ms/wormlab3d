from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from wormlab3d.data.model import MFParameters
from wormlab3d.data.model.mf_parameters import RENDER_MODES


class MFParametersView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return MFParameters

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'objectid',
                },
            ),
            (
                self.prefix + 'created', {
                    'title': 'Created',
                    'type': 'datetime',
                },
            ),
            (
                self.prefix + 'depth', {
                    'title': 'Depth',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'depth_min', {
                    'title': 'Depth min',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'n_points_total', {
                    'title': 'Num. points',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'window_size', {
                    'title': 'Window size',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'window_image_diff_threshold', {
                    'title': 'Window diffs',
                    'type': 'scientific',
                    'precision': 1,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'use_master', {
                    'title': 'Use master',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'masks_threshold', {
                    'title': 'Masks threshold',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'render_mode', {
                    'title': 'Render mode',
                    'type': 'enum',
                    'choices': RENDER_MODES
                },
            ),
            (
                self.prefix + 'second_render_prob', {
                    'title': 'Second render prob.',
                    'type': 'float',
                    'precision': 1,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'sigmas_init', {
                    'title': 'Sigmas init',
                    'type': 'float',
                    'precision': 3,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'sigmas_min', {
                    'title': 'Sigmas min',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'sigmas_max', {
                    'title': 'Sigmas max',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'intensities_min', {
                    'title': 'Intensities min',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'curvature_mode', {
                    'title': 'Curvature mode',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'curvature_deltas', {
                    'title': 'Curvature deltas',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'curvature_max', {
                    'title': 'Curvature max',
                    'type': 'float',
                    'precision': 1,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'curvature_relaxation_factor', {
                    'title': 'Curvature rf',
                    'type': 'float',
                    'precision': 1,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'length_min', {
                    'title': 'Length min',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'length_max', {
                    'title': 'Length max',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'length_init', {
                    'title': 'Length init',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'length_warmup_steps', {
                    'title': 'Length warmup',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'dX0_limit', {
                    'title': 'dX0 limit',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'dl_limit', {
                    'title': 'dl limit',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'dk_limit', {
                    'title': 'dk limit',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'dpsi_limit', {
                    'title': 'dpsi limit',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'centre_shift_every_n_steps', {
                    'title': 'Centre shift every n',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'centre_shift_threshold', {
                    'title': 'Centre shift threshold',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'centre_shift_adj', {
                    'title': 'Centre shift adj',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'frame_skip', {
                    'title': 'Frame skip',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'n_steps_init', {
                    'title': 'Num. steps init',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'n_steps_max', {
                    'title': 'Num. steps max',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'convergence_tau_fast', {
                    'title': 'Conv. tau fast',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'convergence_tau_slow', {
                    'title': 'Conv. tau slow',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'convergence_threshold', {
                    'title': 'Conv. threshold',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'convergence_patience', {
                    'title': 'Conv. patience',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'convergence_loss_target', {
                    'title': 'Conv. loss target',
                    'type': 'float',
                    'precision': 1,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'optimise_cam_coeffs', {
                    'title': 'Opt. cam coeffs',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_cam_intrinsics', {
                    'title': 'Opt. cam intrinsics',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_cam_rotations', {
                    'title': 'Opt. cam rotations',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_cam_translations', {
                    'title': 'Opt. cam translations',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_cam_distortions', {
                    'title': 'Opt. cam distortions',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_cam_shifts', {
                    'title': 'Opt. cam shifts',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_sigmas', {
                    'title': 'Opt. sigmas',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_exponents', {
                    'title': 'Opt. exponents',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'optimise_intensities', {
                    'title': 'Opt. intensities',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'loss_masks_metric', {
                    'title': 'Loss masks metric',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'loss_masks_multiscale', {
                    'title': 'Loss masks multiscale',
                    'type': 'boolean',
                },
            ),
            (
                self.prefix + 'loss_masks', {
                    'title': 'Loss masks',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_neighbours', {
                    'title': 'Loss neighbours',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_parents', {
                    'title': 'Loss parents',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_aunts', {
                    'title': 'Loss aunts',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_scores', {
                    'title': 'Loss scores',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_smoothness', {
                    'title': 'Loss smoothness',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_curvature', {
                    'title': 'Loss curvature',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_temporal', {
                    'title': 'Loss temporal',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_intersections', {
                    'title': 'Loss intersections',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'algorithm', {
                    'title': 'Algorithm',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'lr_cam_coeffs', {
                    'title': 'lr cam coeffs',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'lr_points', {
                    'title': 'lr points',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'lr_sigmas', {
                    'title': 'lr sigmas',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'lr_exponents', {
                    'title': 'lr exponents',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'lr_intensities', {
                    'title': 'lr intensities',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
        ])
