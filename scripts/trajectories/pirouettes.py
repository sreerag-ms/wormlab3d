import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tvtk.tools import visual

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.plot_utils import make_box_from_pca_mlab
from wormlab3d.toolkit.util import print_args, to_dict, str2bool
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.manoeuvres import get_manoeuvres
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

# show_plots = True
# save_plots = False
show_plots = False
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot pirouettes.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--min-reversal-frames', type=int, default=25,
                        help='Minimum number of reversal frames to use to identify a manoeuvre.')
    parser.add_argument('--min-reversal-distance', type=float, default=0.,
                        help='Minimum reversal distance to use to identify a manoeuvre.')
    parser.add_argument('--min-forward-frames', type=int, default=25,
                        help='Minimum number of forward frames between reversals.')
    parser.add_argument('--min-forward-distance', type=float, default=0.,
                        help='Minimum forward distance between reversals.')
    parser.add_argument('--manoeuvre-window', type=int, default=100,
                        help='Number of frames to include either side of a detected manoeuvre.')

    parser.add_argument('--max-reversal-interval', type=int, default=125,
                        help='Maximum number of frames allowed between reversals of the same cluster.')
    parser.add_argument('--min-manoeuvres-in-cluster', type=int, default=3,
                        help='Minimum manoeuvres required to qualify as a cluster.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    parser.add_argument('--highlights',
                        type=lambda s: [(int(item.split('-')[0]), float(item.split('-')[1])) for item in s.split(',')],
                        help='Highlight specific examples like --highlights=trial_id-onset,trial_id-onset.')

    # Stats
    parser.add_argument('--plot-nts', type=str2bool, default=True, help='Plot the number of turns.')
    parser.add_argument('--plot-vols', type=str2bool, default=True, help='Plot the volume explored.')
    parser.add_argument('--plot-nonps', type=str2bool, default=True, help='Plot the non-planarity.')
    parser.add_argument('--plot-ts', type=str2bool, default=True, help='Plot the onset times.')

    # 3D plots
    parser.add_argument('--width-3d', type=int, default=1000, help='Width of 3D plot (in pixels).')
    parser.add_argument('--height-3d', type=int, default=1000, help='Height of 3D plot (in pixels).')
    parser.add_argument('--distance', type=float, default=4., help='Camera distance (in worm lengths).')
    parser.add_argument('--azimuth', type=lambda s: [int(item) for item in s.split(',')], default=[70, ],
                        help='Azimuths.')
    parser.add_argument('--elevation', type=lambda s: [int(item) for item in s.split(',')], default=[45, ],
                        help='Elevations.')
    parser.add_argument('--roll', type=lambda s: [int(item) for item in s.split(',')], default=[45, ], help='Rolls.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_rf={args.min_reversal_frames}' \
           f'_rd={args.min_reversal_distance}' \
           f'_ff={args.min_forward_frames}' \
           f'_fd={args.min_forward_distance}' \
           f'_mw={args.manoeuvre_window}' \
           f'_ri={args.max_reversal_interval}' \
           f'_s={args.min_manoeuvres_in_cluster}'


def _nonp(pca: PCA) -> np.ndarray:
    r = pca.explained_variance_ratio_.T
    return r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    metas = ds.metas
    durations = {}
    n_turns = {}
    nonps = {}
    vols = {}
    timings = {}
    ids = {}

    # Group by concentration and then by reconstruction
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in durations:
            durations[c] = []
            n_turns[c] = []
            nonps[c] = []
            vols[c] = []
            timings[c] = []
            ids[c] = []

        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        # Fetch reconstruction and centre
        X, _ = get_trajectory(reconstruction_id=rid, smoothing_window=args.smoothing_window)
        X = X - X.mean(axis=(0, 1), keepdims=True)
        N = X.shape[1]

        # Pick trajectory point
        if args.trajectory_point == -1:
            Xt = X.mean(axis=1)
        else:
            u = round(args.trajectory_point * N)
            if u == N:
                u -= 1
            assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
            Xt = X[:, u]

        # Get the manoeuvres
        manoeuvres = get_manoeuvres(
            X,
            Xt,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            min_forward_frames=args.min_forward_frames,
            min_forward_distance=args.min_forward_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=False,
        )

        # Cluster the manoeuvres
        clusters = []
        active_cluster = []
        for i, m in enumerate(manoeuvres):
            if len(active_cluster) > 0:
                if m['rev_start_idx'] - active_cluster[-1]['rev_end_idx'] < args.max_reversal_interval:
                    active_cluster.append(m)
                else:
                    if len(active_cluster) >= args.min_manoeuvres_in_cluster:
                        clusters.append(active_cluster)
                    active_cluster = [m]
            else:
                active_cluster = [m]

        # Add any final cluster
        if len(active_cluster) >= args.min_manoeuvres_in_cluster:
            clusters.append(active_cluster)
        logger.info(f'Found {len(clusters)} clusters.')

        # Build the pirouette statistics
        for cluster in clusters:
            start_idx = cluster[0]['start_idx']
            end_idx = cluster[-1]['end_idx']

            # Calculate PCA on entire pirouette
            Xp = Xt[start_idx:end_idx]
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            pca.fit(Xp)

            # Calculate volume of PCA-aligned cuboid extents
            nonpc = np.ptp(Xp @ pca.components_, axis=0)
            vol = np.product(nonpc)

            # Collate
            durations[c].append((end_idx - start_idx) / fps)
            n_turns[c].append(len(cluster))
            nonps[c].append(_nonp(pca))
            vols[c].append(vol)
            timings[c].append(start_idx / fps)
            ids[c].append(reconstruction.trial.id)

    # Sort by concentration
    durations = {k: np.array(v) for k, v in sorted(list(durations.items()))}
    n_turns = {k: np.array(v) for k, v in sorted(list(n_turns.items()))}
    nonps = {k: np.array(v) for k, v in sorted(list(nonps.items()))}
    vols = {k: np.array(v) for k, v in sorted(list(vols.items()))}
    timings = {k: np.array(v) for k, v in sorted(list(timings.items()))}
    ids = {k: np.array(v) for k, v in sorted(list(ids.items()))}

    return durations, n_turns, nonps, vols, timings, ids


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    durations = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            durations = {}
            n_turns = {}
            nonps = {}
            vols = {}
            timings = {}
            ids = {}
            for name in data.files:
                c, d_or_a = name.split('_')
                if d_or_a != 'durations':
                    continue
                c = float(c)
                durations[c] = data[name]
                n_turns[c] = data[f'{c}_n-turns']
                nonps[c] = data[f'{c}_nonps']
                vols[c] = data[f'{c}_vols']
                timings[c] = data[f'{c}_timings']
                ids[c] = data[f'{c}_ids']
            durations = {k: v for k, v in sorted(list(durations.items()))}
            n_turns = {k: v for k, v in sorted(list(n_turns.items()))}
            nonps = {k: v for k, v in sorted(list(nonps.items()))}
            vols = {k: v for k, v in sorted(list(vols.items()))}
            timings = {k: v for k, v in sorted(list(timings.items()))}
            ids = {k: v for k, v in sorted(list(ids.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            durations = None
            logger.warning(f'Could not load cache: {e}')

    if durations is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        durations, n_turns, nonps, vols, timings, ids = _calculate_data(args, ds)
        data = {}
        for c, d_vals in durations.items():
            data[f'{c}_durations'] = d_vals
            data[f'{c}_n-turns'] = n_turns[c]
            data[f'{c}_nonps'] = nonps[c]
            data[f'{c}_vols'] = vols[c]
            data[f'{c}_timings'] = timings[c]
            data[f'{c}_ids'] = ids[c]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, durations, n_turns, nonps, vols, timings, ids


def _plot_statistics():
    """
    Plot the pauses, durations against activity.
    """
    args = parse_args()
    ds, durations, n_turns, nonps, vols, timings, ids = _generate_or_load_data(args, rebuild_cache=False,
                                                                               cache_only=False)
    timings = {c: t / 60 for c, t in timings.items()}

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    d_vals = np.concatenate([d for c, d in durations.items() if _is_included(c)])
    nt_vals = np.concatenate([a for c, a in n_turns.items() if _is_included(c)])
    np_vals = np.concatenate([ad for c, ad in nonps.items() if _is_included(c)])
    v_vals = np.concatenate([v for c, v in vols.items() if _is_included(c)])
    t_vals = np.concatenate([t for c, t in timings.items() if _is_included(c)])
    id_vals = np.concatenate([i for c, i in ids.items() if _is_included(c)])

    # Get highlights
    hl_d = []
    hl_nt = []
    hl_np = []
    hl_v = []
    hl_ts = []
    if len(args.highlights) > 0:
        for trial_id, onset in args.highlights:
            locs = id_vals == trial_id
            idx = locs.nonzero()[0][np.argmin(np.abs(d_vals[locs] - onset))]
            hl_d.append(d_vals[idx])
            hl_nt.append(nt_vals[idx])
            hl_np.append(np_vals[idx])
            hl_v.append(v_vals[idx])
            hl_ts.append(t_vals[idx])

    # Set up plot
    plt.rc('axes', labelsize=9)  # fontsize of the X label
    plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=8)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=3)
    plt.rc('ytick.major', pad=2, size=3)

    n_rows = 1 + sum([int(args.plot_nts), int(args.plot_vols), int(args.plot_nonps), int(args.plot_ts)])

    gs = GridSpec(
        nrows=n_rows,
        ncols=2,
        width_ratios=(4, 2.5),
        height_ratios=(2.2,) + ((3,) * (n_rows - 1)),
        wspace=0,
        hspace=0,
        top=0.97,
        bottom=0.12,
        left=0.1,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, n_rows))
    row_idx = 0

    def scatter_args(c_):
        return dict(
            label=f'{c_:.2f}% ({len(durations[c_]):,d})',
            s=20,
            marker='o',
            facecolors='none',
            edgecolors=colours[c_],
            alpha=0.7
        )

    scatter_args_hl = dict(
        s=40,
        marker='o',
        facecolors='none',
        edgecolors='red',
        alpha=0.9
    )

    hist_args = dict(bins=10, density=True, rwidth=0.9)
    legend = None

    # Get colours
    cm = plt.get_cmap('jet')
    concs = [float(k) for k in durations.keys() if _is_included(k)]
    ticks = np.arange(min(concs), max(concs) + 0.25, 0.25)
    all_colours = cm(np.linspace(0, 1, len(ticks)))
    colours = {c: all_colours[i] for i, c in enumerate(ticks) if c in concs}

    def _make_plots(
            y_label: str,
            y_vals: Dict[float, np.ndarray],
            hl_vals: np.ndarray,
            y_ticks: List[float],
            hist_ticks: List[float],
            hist_colour: str,
            hist_bins: int = 10,
    ):
        nonlocal legend
        vals = np.concatenate([v for c, v in y_vals.items() if _is_included(c)])
        ax_scat_ = fig.add_subplot(gs[row_idx + 1, 0])
        ax_scat_.set_ylabel(y_label, labelpad=8)
        for c, durations_c in durations.items():
            if not _is_included(c):
                continue
            ax_scat_.scatter(durations_c, y_vals[c], **scatter_args(c))
        ax_scat_.scatter(hl_d, hl_vals, **scatter_args_hl)
        ax_scat_.set_yticks(y_ticks)
        ax_scat_.spines['top'].set_visible(False)
        ax_scat_.spines['right'].set_visible(False)

        for d in hl_d:
            ax_scat_.axvline(x=d, color=scatter_args_hl['edgecolors'], zorder=-1)

        # Histogram
        ax_hist_ = fig.add_subplot(gs[row_idx + 1, 1])
        ax_hist_.hist(vals, orientation='horizontal', color=hist_colour, bins=hist_bins, density=True, rwidth=0.9)
        ax_hist_.tick_params(axis='y', left=False, labelleft=False)
        ax_hist_.spines['left'].set(linestyle='--', color='grey')
        ax_hist_.set_xticks(hist_ticks)

        # Position-specific bits
        if row_idx == 0:
            legend = ax_scat_.legend()
        if row_idx != n_rows - 2:
            ax_scat_.spines['bottom'].set(linestyle='--', color='grey')
            ax_scat_.tick_params(axis='x', bottom=False, labelbottom=False)
        else:
            ax_scat_.set_xlabel('Duration (s)', labelpad=5)
            ax_hist_.set_xlabel('Density')

    # Number of turns
    if args.plot_nts:
        _make_plots(
            y_label='Number of turns',
            y_vals=n_turns,
            hl_vals=hl_nt,
            y_ticks=[3, 5, 7],
            hist_ticks=[0.5],
            hist_bins=len(np.unique(nt_vals)),
            hist_colour='skyblue'
        )
        row_idx += 1

    # Volumes
    if args.plot_vols:
        _make_plots(
            y_label='Volume (mm$^{-1}$)',
            y_vals=vols,
            hl_vals=hl_v,
            y_ticks=[0, 0.1, 0.2],
            hist_ticks=[10],
            hist_colour='tab:olive'
        )
        row_idx += 1

    # Non-planarity
    if args.plot_nonps:
        _make_plots(
            y_label='Non-planarity',
            y_vals=nonps,
            hl_vals=hl_np,
            y_ticks=[0, 0.15, 0.3],
            hist_ticks=[10],
            hist_colour='plum'
        )
        row_idx += 1

    # Onset times
    if args.plot_ts:
        _make_plots(
            y_label='Onset (mins)',
            y_vals=timings,
            hl_vals=hl_ts,
            y_ticks=[0, 5, 10],
            hist_ticks=[0.15],
            hist_bins=8,
            hist_colour='tan'
        )
        row_idx += 1

    # Duration histogram
    ax_hist_duration = fig.add_subplot(gs[0, 0])
    ax_hist_duration.hist(d_vals, orientation='vertical', color='green', **hist_args)
    ax_hist_duration.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_hist_duration.spines['bottom'].set(linestyle='--', color='grey')
    ax_hist_duration.set_ylabel('Density')
    # ax_hist_duration.set_yticks([0.5])

    # Legend
    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_hist_duration.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1),
                            bbox_transform=ax_hist_duration.transAxes, ncol=2, columnspacing=0.7, handletextpad=0.2)
    legend.remove()


def _plot_pirouette_3d(
        args: Namespace,
        plot_idx: int,
        reconstruction: Reconstruction,
        start_idx: int,
        end_idx: int,
):
    """
    Make a 3D plot of a pirouettes.
    """
    trial = reconstruction.trial
    r_start_frame = max(reconstruction.start_frame_valid, start_idx)
    r_end_frame = min(reconstruction.end_frame_valid, end_idx)

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)
    mp = Xr.mean(axis=(0, 1))
    Xrc = Xr - mp

    # Pick trajectory point and smooth
    if args.trajectory_point == -1:
        Xt = Xr.mean(axis=1)
    else:
        N = Xr.shape[1]
        u = round(args.trajectory_point * N)
        if u == N:
            u -= 1
        assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
        logger.info(f'Using trajectory point {u}/{N}.')
        Xt = Xr[:, u]
    if args.smoothing_window > 1:
        Xt = smooth_trajectory(Xt, window_len=args.smoothing_window)
    Xtc = Xt - mp
    Xpc_smoothed = smooth_trajectory(Xrc, window_len=args.smoothing_window)
    speeds = calculate_speeds(Xpc_smoothed, signed=True) * trial.fps

    # Get curvature
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=5)
    K = np.abs(Z)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        lengths = ts.get('length', r_start_frame, r_end_frame + 1)[:, 0]
    else:
        lengths = np.linalg.norm(Xrc[:, 1:] - Xrc[:, :-1], axis=-1).sum(axis=-1)

    # Construct trajectory colours
    cmap = plt.get_cmap('PRGn')
    vmax = np.abs(speeds).max()
    vmin = -vmax
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Set up mlab figure
    fig = mlab.figure(size=(args.width_3d * 2, args.height_3d * 2), bgcolor=(1, 1, 1))
    visual.set_viewer(fig)

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 64
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Render the trajectory with simple lines
    path = mlab.plot3d(*Xtc.T, speeds, vmax=vmax, vmin=vmin, opacity=0.8, tube_radius=None, line_width=8)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Add the cuboid
    make_box_from_pca_mlab(
        X=Xtc,
        colour='aqua',
        outline_colour='teal',
        dimensions='extents',
        opacity=0.1,
        draw_outline=True,
        outline_opacity=0.8,
        outline_tube_radius=0.001,
        fig=fig,
    )

    # Select postures
    dists_head = cdist(Xpc_smoothed[:, 0], Xtc)
    dists_tail = cdist(Xpc_smoothed[:, -1], Xtc)
    dists = np.max(np.stack([np.min(dists_head, axis=1), np.min(dists_tail, axis=1)]), axis=0)
    posture_idxs, _ = find_peaks(dists, distance=len(Xtc) / 5, prominence=0.01, height=lengths.mean() * 0.2)

    for posture_idx in posture_idxs:
        # Set up the artist and add the pieces
        NF = NaturalFrame(Xrc[posture_idx])
        fa = FrameArtistMLab(
            NF,
            use_centred_midline=False,
            surface_opts={'radius': 0.025 * lengths.mean()},
            mesh_opts={'opacity': 0.25}
        )
        fa.add_surface(fig, v_min=-K.max(), v_max=K.max())

    # Focus on the middle of the manoeuvre
    centre = Xtc.min(axis=0) + np.ptp(Xtc, axis=0) / 2

    # Get aspect
    aspect = {}
    for k in ['azimuth', 'elevation', 'roll']:
        if len(getattr(args, k)) > plot_idx:
            aspect[k] = getattr(args, k)[plot_idx]
        else:
            aspect[k] = getattr(args, k)[0]

    # Draw plot
    mlab.view(
        figure=fig,
        distance=args.distance,
        focalpoint=centre,
        **aspect
    )

    # # Useful for getting the view parameters when recording from the gui:
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [0.49475351657919076, -2.548112060397567, 2.9932712449202983]
    # scene.scene.camera.focal_point = [-0.0009029966572986492, 0.031932146749383494, -0.02296755697415165]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.9454018697358127, 0.3075819060897899, 0.1077435647554015]
    # scene.scene.camera.clipping_range = [2.9464434264318697, 5.5190396232517145]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    return fig


def plot_pirouettes(
        plot_stats: bool,
        plot_3d: bool,
):
    """
    Make 3D plots of some pirouettes.
    """
    args = parse_args()
    ds, durations, n_turns, nonps, vols, timings, ids = _generate_or_load_data(args, rebuild_cache=False,
                                                                               cache_only=False)
    d_vals = np.concatenate([d for c, d in durations.items()])
    t_vals = np.concatenate([t for c, t in timings.items()])
    id_vals = np.concatenate([i for c, i in ids.items()])

    if save_plots:
        # Copy the spec with final args to the output dir
        output_dir = LOGS_PATH / (f'{START_TIMESTAMP}'
                                  f'_pirouettes'
                                  f'_{_identifiers(args)}'
                                  f'_n={len(d_vals)}')
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / 'spec.yml', 'w') as f:
            yaml.dump(to_dict(args), f)

    # Plot stats
    if plot_stats:
        _plot_statistics()
        if save_plots:
            path = output_dir / f'stats.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()

    # Plot highlights
    if plot_3d and len(args.highlights) > 0:
        for i, (trial_id, onset) in enumerate(args.highlights):
            for rid in ds.metas['reconstruction']:
                reconstruction = Reconstruction.objects.get(id=rid)
                if reconstruction.trial.id == trial_id:
                    break
            assert reconstruction.trial.id == trial_id
            fps = reconstruction.trial.fps
            locs = id_vals == trial_id
            idx = locs.nonzero()[0][np.argmin(np.abs(d_vals[locs] - onset))]
            fig = _plot_pirouette_3d(
                plot_idx=i,
                args=args,
                reconstruction=reconstruction,
                start_idx=int(t_vals[idx] * fps),
                end_idx=int((t_vals[idx] + d_vals[idx]) * fps),
            )
            if save_plots:
                path = output_dir / f'trial={trial_id}_onset={onset}_frame={idx}.png'
                logger.info(f'Saving 3D plot to {path}.')
                fig.scene._lift()
                img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                img.save(path)
                mlab.clf(fig)
                mlab.close()
                logger.info(f'Saving plot to {path}.')
            if show_plots:
                mlab.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if show_plots:
        interactive()
    else:
        mlab.options.offscreen = True

    plot_pirouettes(
        plot_stats=True,
        plot_3d=True,
    )

    # 76-56,110-45,103-27
