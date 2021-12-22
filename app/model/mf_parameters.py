from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from wormlab3d.data.model import MFParameters


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
                self.prefix + 'use_master', {
                    'title': 'Use master',
                    'type': 'boolean',
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
                self.prefix + 'masks_threshold', {
                    'title': 'Masks threshold',
                    'type': 'float',
                    'precision': 2,
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
                self.prefix + 'loss_sigmas', {
                    'title': 'Loss sigmas',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
            (
                self.prefix + 'loss_intensities', {
                    'title': 'Loss intensities',
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
                self.prefix + 'loss_temporal', {
                    'title': 'Loss temporal',
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
                self.prefix + 'lr_intensities', {
                    'title': 'lr intensities',
                    'type': 'scientific',
                    'precision': 1,
                },
            ),
        ])
