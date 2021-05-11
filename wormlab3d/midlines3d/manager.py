from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import gridspec
from torch import Tensor, nn
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE, N_WORM_POINTS, CAMERA_IDXS
from wormlab3d.data.model import SegmentationMasks, NetworkParameters
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.midlines3d.args.network_args import Midline3DNetworkArgs, ENCODING_MODE_DELTA_VECTORS, \
    ENCODING_MODE_DELTA_ANGLES, ENCODING_MODE_DELTA_ANGLES_BASIS, MAX_DECAY_FACTOR
from wormlab3d.midlines3d.args.runtime_args import Midline3DRuntimeArgs
from wormlab3d.midlines3d.data_loader import get_data_loader
from wormlab3d.midlines3d.enc_dec import EncDec
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.nn.args import OptimiserArgs
from wormlab3d.nn.ema import EMA
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.util import to_numpy, is_bad


class Manager(BaseManager):
    def __init__(
            self,
            runtime_args: Midline3DRuntimeArgs,
            dataset_args: DatasetSegmentationMasksArgs,
            net_args: Midline3DNetworkArgs,
            optimiser_args: OptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

        # Register exponential moving averages
        ema = EMA()
        ema.register('loss', decay=0.9)
        ema.register('reg', decay=0.9)
        ema.register('loss_g', decay=0.9)
        ema.register('real_validity', decay=0.9)
        ema.register('recursion_depth_ema', decay=0.99)
        ema.register('use_approx', decay=0.99, val=-1)
        ema.register('r_ema.99', decay=0.99)
        ema.register('r_ema.999', decay=0.999)
        ema.register('total_loss_ema.99', decay=0.99)
        ema.register('total_loss_ema.999', decay=0.999)
        ema.register('total_loss_grad_ema', decay=0.9)
        ema.register('decay_factor', decay=0.99, val=self.net.decay_factor)
        self.ema = ema

        self.metric_keys.append('loss/d')
        self.metric_keys.append('recursion_depth')
        self.metric_keys.append('loss/approx')
        self.metric_keys.append('use_approx')

    @property
    def input_shape(self) -> Tuple[int]:
        """A triplet of segmentation maps, one from each camera."""
        return (3,) + PREPARED_IMAGE_SIZE

    @property
    def output_shape(self) -> Tuple[int]:
        """
        All outputs to be found along the channel dimension. Spatial dimensions collapsed to 1x1.
        Depending on encoding mode, the data in the channel dimension is:
         - ALL: Offset coordinates to centre the worm in each camera view. (3x2)
         - ENCODING_MODE_DELTA_VECTORS: Scaled e0 vectors defining the tangent to the curve. (Nx3)
         - ENCODING_MODE_DELTA_ANGLES | ENCODING_MODE_DELTA_ANGLES_BASIS:
            theta0, phi0 defining the initial spherical angles at the head point. (4)
         - ENCODING_MODE_DELTA_ANGLES: Delta-angles to be cumulatively summed to the initial angles to form the curve. (Nx2)
         - ENCODING_MODE_DELTA_ANGLES_BASIS: Fourier curve basis coefficients from which to generate the delta-angles. (Bx2)
        """

        # All encoding modes need 3 xy vectors to centre the worm in each camera view
        c_out = 3 * 2

        if self.net_args.encoding_mode == ENCODING_MODE_DELTA_VECTORS:
            # e0 vectors directly produced by network
            c_out += N_WORM_POINTS * 3

        else:
            # theta0, phi0 generated using double-angles
            c_out += 4

            if self.net_args.encoding_mode == ENCODING_MODE_DELTA_ANGLES:
                # delta-angles for incrementing along the body from i.c.
                c_out += (N_WORM_POINTS - 1) * 2
            elif self.net_args.encoding_mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
                # Amplitude and phase coefficients for cosine basis functions to produce delta angles for phi and theta
                c_out += self.net_args.n_basis_fns * 2 * 2

        # Spatial dimensions should be collapsed
        return c_out, 1, 1

    def _generate_dataset(self):
        return generate_masks_dataset(self.dataset_args)

    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        return get_data_loader(
            ds=self.ds,
            ds_args=self.dataset_args,
            train_or_test=train_or_test,
            batch_size=self.runtime_args.batch_size
        )

    def _init_network(self) -> Tuple[BaseNet, NetworkParameters]:
        """
        The network parameters refer to the encoder part of the network.
        Here we wrap this in an encoder-decoder-type model with the decoder being a combination of
        the camera model to project to 2D points and a points-to-masks generator.
        """
        net, net_params = super()._init_network()
        full_net = EncDec(
            net=net,
            blur_sigma=self.runtime_args.reprojection_blur_sigma,
            distorted_cameras=True,
            mode=self.net_args.encoding_mode,
            n_basis_fns=self.net_args.n_basis_fns,
        )

        return full_net, net_params

    def _train_batch(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Train on a single batch of data.
        """
        self.net.train()
        X_in, masks_docs, cam_coeffs_base, points_3d_base, points_2d_base = data
        X_in = X_in.to(self.device)
        cam_coeffs_base = cam_coeffs_base.to(self.device)
        points_3d_base = points_3d_base.to(self.device)
        points_2d_base = points_2d_base.to(self.device)

        recursion_max_depth = 1
        outputs_d = []
        metrics_d = []
        losses_d = []

        loss_weights = {
            'distances': 1,
            # 'd': 1,  # 1
            # 'r': 0.05,  # 10
            # 'kl': 0.02,  # ~ 1
            # 'r': 1,  # 20
            'kl': 0.1,  # ~ 5
            'angles': 0.1,  # 4e-2
            # 'approx': 1,
        }

        # Use the approximation route
        if self.ema['use_approx'] > 0:
            self.net.set_decoder_mode('approx')
        else:
            self.net.set_decoder_mode('original')

        # Update decay factor
        self.net.set_decay_factor(self.ema['decay_factor'])

        # Recursively re-encode the reconstructed inputs as long as the discriminator output thinks it is real
        fooled = True  # Does the discriminator think this is genuine?
        depth = 0
        X_original = X_in.detach()
        while fooled and depth < recursion_max_depth:
            X_out, outputs, metrics = self._process_batch(X_in, cam_coeffs_base, points_3d_base, points_2d_base,
                                                          depth=depth, X_original=X_original)
            outputs_d.append(outputs)
            metrics_d.append(metrics)

            # losses
            d_prob = metrics['d_prob'].detach()
            loss_d = 0
            for k, w in loss_weights.items():
                v = metrics[f'loss/{k}']
                sf = 1  # if k != 'd' else 0.001
                # if k in ['d', 'approx']:
                #     sf = 1
                # elif d_prob < 0.4:
                #     sf = 0
                # else:
                #     sf = 1  #(d_prob - 0.4) * 5/3
                # sf = 1
                v_scaled = w * v * sf  # / (depth + 1)
                metrics[f'loss/{k}/scaled'] = v_scaled
                loss_d = loss_d + v_scaled
            losses_d.append(loss_d)

            fooled = torch.bernoulli(d_prob)
            depth += 1
            X_in = X_out.detach()

            # if self.checkpoint.epoch < 2:
            #     fooled = False

        loss_total = sum(losses_d) / depth
        assert not is_bad(loss_total)

        # Clip gradients
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=100)

        # Calculate gradients and do optimisation step
        self.optimiser.zero_grad()
        loss_total.backward()
        self.optimiser.step()

        with torch.no_grad():
            stats = {
                'recursion_depth': depth,
                'recursion_depth_ema': self.ema('recursion_depth_ema', depth)
            }

            # Collate metrics
            for k in list(metrics_d[0].keys()):
                v_sum = 0
                for d in range(depth):
                    v = metrics_d[d][k]
                    v_sum = v_sum + v
                    # stats[f'{k}/d={d}'] = v
                v_mean = v_sum / depth
                stats[k] = v_mean

            use_approx_threshold = 100
            stats['use_approx'] = self.ema('use_approx', use_approx_threshold - stats['loss/approx'])

            # If loss is within 1.1 of best loss then decrease the decay factor, else increase..?
            stats['loss/total_loss/ema.99'] = self.ema('total_loss_ema.99', loss_total)
            stats['loss/total_loss/ema.999'] = self.ema('total_loss_ema.999', loss_total)
            stats['loss/total_loss/grad_ema'] = self.ema('total_loss_grad_ema', stats['loss/total_loss_ema.99'] - stats[
                'loss/total_loss_ema.999'])
            self.net.best_loss = min(self.net.best_loss, stats['loss/total_loss_ema.999'])
            stats['loss/total_loss/best'] = self.net.best_loss
            if stats['loss/total_loss/best'] < self.net.best_loss * 1.1:
                df = -1
            else:
                df = 1
            ndf = max(0, min(MAX_DECAY_FACTOR, self.ema['decay_factor'] + df))
            stats['decay_factor'] = self.ema('decay_factor', ndf)

            # ------- Log losses -------
            self.tb_logger.add_scalar('batch/train/total_loss', loss_total, self.checkpoint.step)
            for key, val in stats.items():
                self.tb_logger.add_scalar(f'batch/train/{key}', val, self.checkpoint.step)

            # Calculate L2 loss
            norms = self.net.calc_norms()
            weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
            for _, norm in norms.items():
                weights_cumulative_norm += norm
            assert not is_bad(weights_cumulative_norm)
            self.tb_logger.add_scalar('batch/train/w_norm', weights_cumulative_norm.item(), self.checkpoint.step)

        # Increment global step counter
        self.checkpoint.step += 1
        self.checkpoint.examples_count += self.runtime_args.batch_size

        return outputs_d, loss_total, stats

    def _process_batch(
            self,
            X_in,
            cam_coeffs_base,
            points_3d_base,
            points_2d_base,
            depth=0,
            X_original=None
    ):
        outputs = self.net(X_in, cam_coeffs_base, points_3d_base, points_2d_base)
        disc_out, cam_coeffs, points_3d, points_2d, points_2d_opt, X_out, X_opt, mus, log_vars, e0s_scaled, delta_angles, X_approx = outputs

        # Calculate pixel-wise losses (ie, MSE)
        if self.net.use_approx:
            X_ = X_approx
        else:
            X_ = X_out
        r_loss, metrics = self.calculate_losses(X_, X_in)
        if X_original is not None:
            r_loss_og, metrics_og = self.calculate_losses(X_, X_original)
            r_loss = (r_loss + 2 * r_loss_og) / 3
        metrics['loss/r'] = r_loss

        # Losses to propagate are the sum of euclidean distances between the 2d points and their "optimals"
        point_distances = torch.norm(points_2d - points_2d_opt, p=2, dim=-1)
        decay = torch.exp(-torch.arange(N_WORM_POINTS, device=self.device) / N_WORM_POINTS * self.ema['decay_factor'])
        dist_loss = (point_distances * decay).sum(dim=(1, 2)).mean()
        metrics['loss/distances'] = dist_loss

        # Discriminator loss
        disc_out_mean = disc_out.mean()
        d_loss = -disc_out_mean if depth == 0 else disc_out_mean
        d_prob = torch.sigmoid(disc_out_mean)
        metrics['loss/d'] = torch.sigmoid(d_loss)
        metrics['d_prob'] = d_prob

        # Calculate KL-divergence from the stochastic sampling of the distribution
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())
        metrics['loss/kl'] = kl_loss

        # Penalise large delta-angles, these lie in (-1, +1) * max curvature
        angles_loss = (delta_angles**2).sum(dim=(1, 2)).mean()
        metrics['loss/angles'] = angles_loss

        # Approximator
        approx_loss = F.mse_loss(X_approx, X_out, reduction='sum')
        approx_loss = approx_loss / len(X_approx)  # return loss per-datum so different batch sizes can be compared
        metrics['loss/approx'] = approx_loss

        # Track distribution metrics
        metrics['mus/mean'] = mus.mean()
        metrics['mus/var'] = mus.var()
        metrics['log_vars/mean'] = log_vars.mean()
        metrics['log_vars/var'] = log_vars.var()

        # Track shifts
        for i in range(3):
            v = self.net.shifts
            metrics[f'shifts_{i}/mean'] = v[:, i].mean()
            metrics[f'shifts_{i}/var'] = v[:, i].var()
            # metrics[f'shifts/var'] = v.var(dim=1).mean()

        # Track basis coefficients
        if self.net.mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
            for angle in ['phi', 'theta']:
                for Ap in ['A', 'p']:
                    k = f'basis_{angle}_{Ap}'
                    v = getattr(self.net, k)
                    metrics[k + '/mean'] = v.mean()
                    metrics[k + '/var'] = v.var()

        # Track basis coefficients
        if self.net.mode in [ENCODING_MODE_DELTA_ANGLES, ENCODING_MODE_DELTA_ANGLES_BASIS]:
            v = self.net.pre_angles
            for i in range(4):
                angle = 'theta0' if i < 2 else 'phi0'
                metrics[f'angles/{angle}/pre/{i}/mean'] = v[:, i].mean()
                metrics[f'angles/{angle}/pre/{i}/var'] = v[:, i].var()
            for angle in ['theta0', 'phi0']:
                v = getattr(self.net, angle)
                metrics[f'angles/{angle}/mean'] = v.mean()
                metrics[f'angles/{angle}/var'] = v.var()

        return X_, outputs, metrics

    def _make_plots(
            self,
            data: Tuple[Tensor, List[SegmentationMasks], Tensor, Tensor, Tensor],
            outputs_d: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
            train_or_test: str,
            end_of_epoch: bool = False
    ):
        """
        Generate some example plots.
        """
        if self.runtime_args.plot_n_examples > 0 and (
                end_of_epoch or
                (self.runtime_args.plot_every_n_batches > -1
                 and (self.checkpoint.step + 1) % self.runtime_args.plot_every_n_batches == 0)
        ):
            # Select the idxs to plot
            n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
            idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)

            # Unpack variables
            X_in, masks_docs, cam_coeffs_base, points_3d_base, points_2d_base = data

            for depth, outputs in enumerate(outputs_d):
                disc_out, cam_coeffs, points_3d, points_2d, points_2d_opt, X_out, X_opt, mus, log_vars, e0s_scaled, delta_angles, X_approx = outputs
                if depth > 0:
                    X_in = X_prev
                X_prev = X_out

                # Make plots
                suffix = train_or_test + f'_d={depth}'
                # self._plot_3d_midlines(points_3d, masks_docs, suffix, idxs)
                self._plot_masks(X_in, X_out, masks_docs, suffix, idxs, points_2d=points_2d)
                self._plot_masks(X_in, X_opt, masks_docs, suffix + '_opt', idxs, points_2d=points_2d_opt)
                # self._plot_masks(X_out, X_approx, masks_docs, suffix + '_approx', idxs, points_2d=points_2d)
                # self._plot_coefficients(masks_docs, disc_X_in=disc_out, disc_X_out=None, train_or_test=suffix,
                #                         idxs=idxs)

    def _plot_3d_midlines(
            self,
            points_3d: Tensor,
            mask_docs: List[SegmentationMasks],
            train_or_test: str,
            idxs: List[int]
    ):
        """
        Plot generated 3D midline curves.
        """
        points_3d = to_numpy(points_3d)
        n_examples = len(idxs)

        fig = plt.figure(figsize=(n_examples * 4, 6))
        gs = gridspec.GridSpec(
            1, n_examples,
            wspace=0.05,
            hspace=-.99,
            width_ratios=[1] * n_examples,
            top=0.8,
            bottom=0.01,
            left=0.02,
            right=0.98
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}'
        )

        # Colourmap / facecolors
        cmap = plt.cm.get_cmap('plasma')
        fc = cmap((np.arange(N_WORM_POINTS) + 0.5) / N_WORM_POINTS)

        # Scatter plot options
        midline_opts = {
            's': 20,
            'alpha': 0.9,
            'depthshade': True,
            'c': fc
        }

        for i, idx in enumerate(idxs):
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial

            # Prepped image triplet with overlaid input segmentation masks
            ax = fig.add_subplot(gs[0, i], projection='3d')
            vids = "\n\t".join(trial.videos)
            ax.text2D(
                0, 1,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n',
                transform=ax.transAxes
            )

            # Scatter plot of midline
            X = points_3d[idx].transpose().copy()
            ax.scatter(X[0], X[1], X[2], **midline_opts)

        self.tb_logger.add_figure(f'3d_midlines_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)

    def _plot_masks(
            self,
            X_in: Tensor,
            X_out: Tensor,
            mask_docs: List[SegmentationMasks],
            train_or_test: str,
            idxs: List[int],
            points_2d: Tensor = None
    ):
        """
        Plot the segmentation masks and squared errors.
        """
        X_in, X_out = to_numpy(X_in), to_numpy(X_out)
        if points_2d is not None:
            points_2d = to_numpy(points_2d)
            # Colourmap / facecolors
            cmap = plt.cm.get_cmap('plasma')
            fc = cmap((np.arange(N_WORM_POINTS) + 0.5) / N_WORM_POINTS)

            # Scatter plot options
            p2d_opts = {
                's': 1,
                'alpha': 0.6,
                'c': fc
            }
        n_examples = len(idxs)

        fig, axes = plt.subplots(
            nrows=3,
            ncols=n_examples,
            figsize=(n_examples * 5, 7),
            gridspec_kw=dict(
                wspace=0.01,
                hspace=-.99,
                width_ratios=[1] * n_examples,
                top=0.8,
                bottom=0.01,
                left=0.02,
                right=0.98
            ),
            constrained_layout=True,
            squeeze=False
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}, '
            f'blur_sigma={self.runtime_args.reprojection_blur_sigma:.1f}'
        )

        # Calculate squared pixel errors
        errors = np.square(X_in[idxs] - X_out[idxs])

        for i, idx in enumerate(idxs):
            # print('\n\nidx=',idx)
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial
            images = mask_doc.get_images()

            # Stitch images and masks together
            image_triplet = np.concatenate(images, axis=1)
            masks_in_triplet = np.concatenate(X_in[idx], axis=1)
            masks_out_triplet = np.concatenate(X_out[idx], axis=1)
            error_triplet = np.concatenate(errors[i], axis=1)

            # Prepped image triplet with overlaid input segmentation masks
            ax = axes[0, i]
            vids = "\n\t".join(trial.videos)
            ax.text(
                -1, 0,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n'
            )

            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = masks_in_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(masks_in_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if i == 0:
                ax.text(-0.05, 0.1, 'Original+Masks_In', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Error between target and output
            ax = axes[1, i]
            m = ax.imshow(error_triplet, cmap=plt.cm.Reds, vmin=errors.min(), vmax=errors.max())
            if i == 0:
                ax.text(-0.05, 0.4, 'Error', transform=ax.transAxes, rotation='vertical')
            if i == n_examples - 1:
                fig.colorbar(m, ax=ax, format='%.3f', shrink=0.7)
            ax.axis('off')

            # Prepped image triplet with overlaid output segmentation masks
            ax = axes[2, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = masks_out_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(masks_out_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if points_2d is not None:
                for c in CAMERA_IDXS:
                    p2d = points_2d[idx][c]
                    p2d[:, 0] += c * 200
                    ax.scatter(p2d[:, 0], p2d[:, 1], **p2d_opts)
            if i == 0:
                ax.text(-0.05, 0.1, 'Original+Masks_Out', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)

    def _plot_coefficients(
            self,
            mask_docs: List[SegmentationMasks],
            disc_X_in: Tensor,
            disc_X_out: Tensor,
            train_or_test: str,
            idxs: List[int]
    ):
        """
        Plot the basis coefficients
        """

        n_examples = len(idxs)
        basis_phi_A = to_numpy(self.net.basis_phi_A)
        basis_phi_p = to_numpy(self.net.basis_phi_p)
        basis_theta_A = to_numpy(self.net.basis_theta_A)
        basis_theta_p = to_numpy(self.net.basis_theta_p)
        delta_phis = to_numpy(self.net.delta_phis)
        delta_thetas = to_numpy(self.net.delta_thetas)
        phis = to_numpy(self.net.phis)
        thetas = to_numpy(self.net.thetas)

        fig, axes = plt.subplots(
            nrows=8,
            ncols=n_examples,
            figsize=(n_examples * 5, 17),
            gridspec_kw=dict(
                wspace=0.2,
                hspace=0.4,
                top=0.87,
                bottom=0.02,
                left=0.05,
                right=0.95
            ),
            squeeze=False
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}'
        )
        indices_bases = np.arange(self.net.n_basis_fns)
        indices_worm = np.arange(N_WORM_POINTS)

        for i, idx in enumerate(idxs):
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial

            # Plot amplitudes
            ax = axes[0, i]
            vids = "\n\t".join(trial.videos)
            ax.text(
                0, 1.2,
                f'Trial: {trial.id}\n' +
                f'Videos: \n\t{vids}\n' +
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n' +
                (f'Discriminator (real): {disc_X_in[idx].item():.4f}\n' if disc_X_in is not None else '') +
                (f'Discriminator (reconstructed): {disc_X_out[idx].item():.4f}\n' if disc_X_out is not None else ''),
                transform=ax.transAxes
            )

            # Phi amplitudes
            ax.set_title('$\phi$ amplitudes')
            ax.bar(indices_bases, basis_phi_A[idx], align='center')
            ax.set_xticks(indices_bases)
            ax.axhline(y=0, color='gray', linestyle='--')
            if i != 0:
                ax.sharey(axes[0, 0])
                # ax.get_shared_y_axes().join(ax, axes[0, 0])

            # Phi phases
            ax = axes[1, i]
            ax.set_title('$\phi$ phases')
            ax.bar(indices_bases, basis_phi_p[idx], align='center')
            ax.set_xticks(indices_bases)
            ax.axhline(y=0, color='gray', linestyle='--')
            if i != 0:
                ax.sharey(axes[1, 0])
                # ax.get_shared_y_axes().join(ax, axes[1, 0])

            # Phi deltas
            ax = axes[2, i]
            ax.set_title('$\phi$ deltas')
            ax.bar(indices_worm[1:], delta_phis[idx][1:], align='center')
            # ax.set_xticks(indices_worm)
            ax.axhline(y=0, color='gray', linestyle='--')
            if i != 0:
                ax.sharey(axes[2, 0])
                # ax.get_shared_y_axes().join(ax, axes[2, 0])

            # Phis
            ax = axes[3, i]
            ax.set_title('$\phi$')
            ax.bar(indices_worm, phis[idx], align='center')
            # ax.set_xticks(indices_worm)
            ax.axhline(y=0, color='gray', linestyle='--')
            if i != 0:
                ax.sharey(axes[3, 0])
                # ax.get_shared_y_axes().join(ax, axes[3, 0])

            # Theta amplitudes
            ax = axes[4, i]
            ax.set_title('$\\theta$ amplitudes')
            ax.bar(indices_bases, basis_theta_A[idx], align='center')
            ax.set_xticks(indices_bases)
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.sharey(axes[0, 0])
            # ax.get_shared_y_axes().join(ax, axes[0, 0])

            # Theta phases
            ax = axes[5, i]
            ax.set_title('$\\theta$ phases')
            ax.bar(indices_bases, basis_theta_p[idx], align='center')
            ax.set_xticks(indices_bases)
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.sharey(axes[1, 0])
            # ax.get_shared_y_axes().join(ax, axes[1, 0])

            # Theta deltas
            ax = axes[6, i]
            ax.set_title('$\\theta$ deltas')
            ax.bar(indices_worm[1:], delta_thetas[idx][1:], align='center')
            # ax.set_xticks(indices_worm)
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.sharey(axes[2, 0])
            # ax.get_shared_y_axes().join(ax, axes[2, 0])

            # Thetas
            ax = axes[7, i]
            ax.set_title('$\\theta$')
            ax.bar(indices_worm, thetas[idx], align='center')
            # ax.set_xticks(indices_worm)
            ax.axhline(y=0, color='gray', linestyle='--')
            # ax.get_shared_y_axes().join(ax, axes[3, 0])
            ax.sharey(axes[3, 0])

        axes[0, 0].autoscale()
        axes[1, 0].autoscale()
        axes[2, 0].autoscale()
        axes[3, 0].autoscale()

        self.tb_logger.add_figure(f'basis_coeffs_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)
