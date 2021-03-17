import copy
import hashlib
import json
import math
import os
import time
from datetime import timedelta

import numpy as np
import scipy.io as sio
import torch
from mongoengine import DoesNotExist
from torch.utils.data import Dataset as DatasetTorch
from torchvision import transforms

# from inv.sw_api import SimpleWormAPI
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.trajectory import Trajectory
from wormlab3d.data.model.trajectory_dataset import TrajectoryDataset
from wormlab3d.data.model.trial import Trial
from wormlab3d.toolkit.worm import Worm
from wormlab3d.toolkit.eigenworm import WormEigenbasis
from wormlab3d.toolkit.util import hash_data

HOME_DIR = os.path.expanduser('~')
DATA_DIR = HOME_DIR + '/projects/wormpy/data'

# DATA_DIR = '../data'
RAW_DATA_DIR = DATA_DIR + '/raw'
DATASETS_DIR = DATA_DIR + '/datasets'

MAT_PATH = RAW_DATA_DIR + '/XYZBCIF0_PCA_20201117.mat'
DATA_PATH_METADATA = MAT_PATH + '.metadata.npy'
DATA_PATH_XYZ = MAT_PATH + '.xyz.npy'
DATA_PATH_BISHOP = MAT_PATH + '.bishop.npy'
DATA_PATH_CPCA = MAT_PATH + '.cpca'


WORM_LENGTH = 128

META_LABEL_IDX = 0
META_CONC_IDX = 1
META_CLIPID_IDX = 2
META_FRAMENUM_IDX = 3
MIRROR_LABEL_OFFSET = 100

FRAME_DURATION = 1 / 25

eps = 1e-8

LABELS_UNMIRRORED = {
    0: 'Unlabelled',
    1: 'Crawling+',
    -1: 'Crawling-',
    2: 'Turn+',
    -2: 'Turn-',
    3: 'CW coiling+',
    -3: 'CW coiling-',
    4: 'CCW coiling+',
    -4: 'CCW coiling-',
    5: 'Infinity+',
    -5: 'Infinity-'
}

LABELS_MIRRORED = {
    c + MIRROR_LABEL_OFFSET: f'm({l})' for c, l in LABELS_UNMIRRORED.items()
}

LABELS = {**LABELS_UNMIRRORED, **LABELS_MIRRORED}

INVERSE_RESULTS_KEYS = ['e10', 'e20', 'alpha_pref', 'beta_pref', 'gamma_pref', 'XO', 'E1O', 'E2O']


def get_param_hash(inv_opt_params):
    p = inv_opt_params.copy()
    if 'batch_size' in p:
        del p['batch_size']
    return hash_data(p)


class TrajectoryDatasetTorch(DatasetTorch):
    def __init__(
            self,
            ds: TrajectoryDataset,
            augment=False,
            transform=None
    ):
        # self.inverse_data = inverse_data
        self.labels = meta[:, META_LABEL_IDX].astype(np.long)

        # Convert labels to class idxs
        if restrict_classes is not None:
            label_opts = restrict_classes
        else:
            label_opts = list(LABELS_UNMIRRORED.keys())
        if include_mirrors:
            mlo = [l + MIRROR_LABEL_OFFSET for l in label_opts]
            label_opts = label_opts + mlo
        self.label_options = label_opts

        # self.label_options = list(LABELS.keys()) if not restrict_classes else restrict_classes
        self.n_classes = len(self.label_options)
        self.labels_mapped = np.zeros_like(self.labels)
        for i, lbl in enumerate(self.label_options):
            self.labels_mapped[self.labels == lbl] = i

        self.augment = augment
        self.transform = transform

    def __getitem__(self, index):
        item = self.data[index]
        label = self.labels_mapped[index]

        if self.transform is not None:
            item = self.transform(item)

        # Prepare inverse if available
        if self.inverse_data is not None:
            irs = {}
            for k in INVERSE_RESULTS_KEYS:
                ir = self.inverse_data[k][index]
                if self.transform is not None:
                    ir = self.transform(ir)
                irs[k] = ir

            if 1:
                centre = item.mean(axis=(1, 2), keepdims=True)
                item = item - centre

                centre = irs['XO'].mean(axis=(1, 2), keepdims=True)
                irs['XO'] = irs['XO'] - centre
        else:
            irs = None


        return item, label, irs

    def __len__(self):
        return len(self.data)

    def _get_transforms(self):
        transforms_list = []
        if self.augment:
            pass
        transforms_list += [
            transforms.ToTensor(),
        ]

        return transforms.Compose(transforms_list)

    @property
    def input_shape(self):
        return self.frame_shape + (self.n_frames,)

    @property
    def n_samples_per_class(self):
        return torch.tensor(
            [(self.labels_mapped == i).sum() for i in range(self.n_classes)]
        )



class XYZDataset(TrajectoryDatasetTorch):
    @property
    def frame_shape(self):
        return 3, WORM_LENGTH


class XYZInvDataset(XYZDataset):
    @property
    def frame_shape(self):
        return 3, WORM_LENGTH


class BishopDataset(TrajectoryDatasetTorch):
    @property
    def frame_shape(self):
        return 2, WORM_LENGTH


class CPCADataset(TrajectoryDatasetTorch):
    def __init__(self, *args, n_components=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_components = n_components

    @property
    def frame_shape(self):
        return self.n_components * 2, 1


def get_input_shape(data_type, n_frames, n_cpca_components):
    if data_type in ['xyz', 'xyz_inv']:
        return 3, WORM_LENGTH, n_frames
    elif data_type == 'bishop':
        return 2, WORM_LENGTH, n_frames
    elif data_type == 'cpca':
        return 2 * n_cpca_components, 1, n_frames


def extract_mat():
    print(f'Extracting data from {MAT_PATH}...')
    mat = sio.loadmat(MAT_PATH)
    ws = mat['XYZ'].transpose((0, 2, 1))
    N = len(ws)

    xyzs = np.zeros([N, 128, 3], np.float64)
    bishops = np.zeros([N, 128], np.complex64)
    metadata = np.zeros([N, 4], np.float32)

    for n, w in enumerate(ws):
        if n % 100 == 0:
            print(f'{n + 1}/{N}')
        xyz = w[:, :3]
        xyzs[n] = xyz
        bishops[n] = Worm(xyz).to_bishop_()
        metadata[n] = w[0, 3:]

    print(f'Saving xyz data to {DATA_PATH_XYZ}...')
    np.save(DATA_PATH_XYZ, xyzs)
    print(f'Saving bishop data to {DATA_PATH_BISHOP}...')
    np.save(DATA_PATH_BISHOP, bishops)
    print(f'Saving metadata to {DATA_PATH_METADATA}...')
    np.save(DATA_PATH_METADATA, metadata)


def generate_or_load_cpca_projections(n_cpca_components, restrict_length=None, mirrored=False):
    path = DATA_PATH_CPCA + '_' + str(n_cpca_components) \
           + (f'_l{restrict_length}' if restrict_length is not None else '') \
           + ('_m' if mirrored else '') \
           + '.npy'
    try:
        data = np.load(path)
        return data
    except Exception:
        print(f'CPCA({n_cpca_components}) postures not found, generating...')

    bishop = np.load(DATA_PATH_BISHOP)

    if restrict_length is not None:
        bishop = bishop[:, :restrict_length]
        if restrict_length == 43:
            fn = CPCA_PATH_43
        elif restrict_length == 64:
            fn = CPCA_PATH_64
        else:
            raise RuntimeError(f'No CPCA basis file available for worm length {restrict_length}')
    else:
        fn = CPCA_PATH

    if mirrored:
        bishop = np.conj(bishop)

    e = WormEigenbasis(fn, len=n_cpca_components)
    data = []
    for b in bishop:
        data.append(e.inv(b))
    data = np.array(data)
    print(f'Saving CPCA data projections to {path}')
    np.save(path, data)
    return data


def load_extracted_data(data_type, n_cpca_components=1, restrict_length=None, include_mirrors=False):
    try:
        # Load non-mirrored data
        if data_type in ['xyz', 'xyz_inv']:
            data = np.load(DATA_PATH_XYZ)
        elif data_type == 'bishop':
            data = np.load(DATA_PATH_BISHOP)
        elif data_type == 'cpca':
            data = generate_or_load_cpca_projections(
                n_cpca_components,
                restrict_length=restrict_length,
                mirrored=False
            )

        # Restrict worm length
        if data_type != 'cpca' and restrict_length is not None:
            data = data[:, :restrict_length]

        # Load metadata
        meta = np.load(DATA_PATH_METADATA)

        # Include mirror images of the data if needed
        if include_mirrors:
            if data_type in ['xyz', 'xyz_inv']:
                data_m = -data
            elif data_type == 'bishop':
                data_m = np.conj(data)
            elif data_type == 'cpca':
                data_m = generate_or_load_cpca_projections(
                    n_cpca_components,
                    restrict_length=restrict_length,
                    mirrored=True
                )
            data = np.vstack([data, data_m])
            meta_m = np.copy(meta)
            meta_m[:, META_LABEL_IDX] += MIRROR_LABEL_OFFSET
            meta = np.vstack([meta, meta_m])

    except Exception:
        print(f'Could not load extracted data.')
        extract_mat()
        return load_extracted_data(data_type, n_cpca_components, restrict_length, include_mirrors)

    return data, meta


def generate_dataset(
        data_type,
        train_test_split=0.8,
        n_frames=10,
        frame_shift=5,
        n_cpca_components=2,
        restrict_tags=None,
        restrict_concs=None,
        restrict_length=None,
        include_mirrors=False,
        inv_opt_params={}
) -> TrajectoryDataset:

    DS = TrajectoryDataset(
        data_type=data_type,
        train_test_split_target=train_test_split,
        n_frames=n_frames,
        frame_shift=frame_shift,
        n_cpca_components=n_cpca_components if data_type == 'cpca' else None,
        restrict_tags=restrict_tags,
        restrict_concs = restrict_concs,
        restrict_length = restrict_length,
        include_mirrors=include_mirrors,
        inv_opt_params=inv_opt_params
    )

    # # Load all data and metadata for all frames
    # ds, meta = load_extracted_data(
    #     data_type,
    #     n_cpca_components,
    #     restrict_length=restrict_length,
    #     include_mirrors=include_mirrors
    # )
    # print('ds.shape', ds.shape)

    # Build filter  todo: pipeline/aggregation?
    filter = {}

    # Restrict to selected tags
    if restrict_tags is not None:
        filter['tags__contain'] = restrict_tags

    # Restrict to concentrations
    if restrict_concs is not None:
        filter['experiment__concentration__in'] = restrict_concs

    # Fetch matching trials
    trials = Trial.objects.filter(**filter)

    # Train/test split
    ds_train = []
    ds_test = []

    # once,
    # 1. determine the amount of overlap:
    #    for sequence A0, how many (n>=0) of the subsequent sequences An
    #    contain frames also contained in A0?  -> n_overlap
    n_overlap = max(0, n_frames - frame_shift)

    # for each tag,
    # 1. see how many frames total there are
    # 2. set target total train/test frame counts

    tag_info = {}
    for tag_id in restrict_tags:
        n_frames_tag = Frame.objects.filter(
            trial__in=trials,
            tags__contains=tag_id
        ).count()
        target_train_c = int(n_frames_tag * train_test_split)
        target_test_c = n_frames_tag - target_train_c

        tag_info[tag_id] = {
            'n_frames_all': n_frames_tag,
            'n_frames_train': 0,
            'n_frames_test': 0,

            'n_frames_target_train': target_train_c,
            'n_frames_target_test': target_test_c,

            'n_seq_train': 0,
            'n_seq_test': 0,

            'seq_all': [],
            'seq_train': [],
            'seq_test': [],

            'meta_all': [],
            'meta_train': [],
            'meta_test': [],
        }

    # for each contiguous clip of a gait,
    # 1. collect all (overlapping) sequences into sequence groups
    for trial in trials:
        clips = trial.get_clips(
            filter={'tags__contains': restrict_tags}  # todo
        )

        for clip in clips:
            # Collect all trajectories in the clip
            seqs_in_clip = []
            k = 0
            while k + n_frames < len(clip):
                sequence = clip[k:k + n_frames]

                trajectory = Trajectory(dataset=DS)
                trajectory.set_from_sequence(
                    sequence,
                    centre=data_type in ['xyz', 'xyz_inv']
                )

                seqs_in_clip.append(trajectory)
                k += frame_shift

            # Store the groups in the class info dict
            clip_tags = clip[0].tags
            for tag in clip_tags:
                ti = tag_info[tag.id]
                ti['seq_all'].append(seqs_in_clip)

    # For each class, split or assign the clips
    for tag_id, ti in tag_info.items():
        # 1. if n_seq_in_clip >> n_overlap then split, else save for later
        seq_groups_to_split = []
        seq_groups_to_assign = []
        for start_idx, seqs_in_clip in enumerate(ti['seq_all']):
            if len(seqs_in_clip) > n_overlap * 3:
                seq_groups_to_split.append(seqs_in_clip)
            else:
                seq_groups_to_assign.append(seqs_in_clip)

        # 2a: Split clips
        # 1. n_avail_seq_in_clip = n_seq_in_clip - n_overlap
        # 2. determine size of desired train/test sequences sets
        # 3. randomly pick whether to split it by train/test or test/train and determine split point
        # 4. add sequences to the sets and discard the overlap
        for start_idx, seqs_in_clip in enumerate(seq_groups_to_split):
            n_avail_seq_in_clip = len(seqs_in_clip) - n_overlap
            target_n_train_seq = int(n_avail_seq_in_clip * train_test_split)
            target_n_test_seq = n_avail_seq_in_clip - target_n_train_seq
            if n_avail_seq_in_clip == 1:
                if bool(np.random.rand() > 0.5):
                    train_seqs = seqs_in_clip
                    test_seqs = []
                else:
                    train_seqs = []
                    test_seqs = seqs_in_clip
            else:
                if bool(np.random.rand() > 0.5):
                    train_seqs = seqs_in_clip[:target_n_train_seq]
                    test_seqs = seqs_in_clip[-target_n_test_seq:]
                else:
                    train_seqs = seqs_in_clip[-target_n_train_seq:]
                    test_seqs = seqs_in_clip[:target_n_test_seq]
            assert n_avail_seq_in_clip == len(train_seqs) + len(test_seqs)

            ti['seq_train'].extend(train_seqs)
            ti['seq_test'].extend(test_seqs)

            ti['n_seq_train'] += len(train_seqs)
            ti['n_seq_test'] += len(test_seqs)
            ti['n_frames_train'] += n_frames + (len(train_seqs) - 1) * frame_shift
            ti['n_frames_test'] += n_frames + (len(test_seqs) - 1) * frame_shift

        if len(seq_groups_to_assign):
            # 2b: Assign entire clips
            # 1. count numbers of remaining sequences which must be grouped together due to overlap
            # 2. count total number of frames and set target number of train/test sequences
            n_frames_unassigned = 0
            for start_idx, seq_groups in enumerate(seq_groups_to_assign):
                n_frames_unassigned += n_frames + (len(seq_groups) - 1) * frame_shift

            # 3. do X times, take the best:
            #   a) randomly order the sequence groups
            #   b) sum the number of sequences in each group until the test target is reached
            #   c) record number of train/test sequences and where groups where assigned
            #   d) set the score = split target - split actual
            n_trials = 10
            idxs_attempts = [[] for _ in range(n_trials)]
            scores = np.zeros(n_trials)
            for start_idx in range(n_trials):
                idxs = np.random.permutation(len(seq_groups_to_assign))
                n_test_frames = 0
                for k, j in enumerate(idxs):
                    idxs_attempts[start_idx].append(j)
                    n_test_frames += n_frames + (len(seq_groups_to_assign[j]) - 1) * frame_shift
                    if n_test_frames + ti['n_frames_test'] >= ti['n_frames_target_test']:
                        break
                n_train_frames = ti['n_frames_train'] + n_frames_unassigned - n_test_frames
                split_targ = ti['n_frames_target_test'] / ti['n_frames_target_train']
                split_attempt = (ti['n_frames_test'] + n_test_frames) / (ti['n_frames_train'] + n_train_frames + 1e-5)
                scores[start_idx] = (split_attempt - split_targ)**2

            # Assign sequence groups as appropriate
            test_group_idxs = idxs_attempts[np.argmin(scores)]
            for start_idx, seqs_in_clip in enumerate(seq_groups_to_assign):
                if start_idx in test_group_idxs:
                    ti['seq_test'].extend(seqs_in_clip)
                    ti['n_seq_test'] += len(seqs_in_clip)
                    ti['n_frames_test'] += n_frames + (len(seqs_in_clip) - 1) * frame_shift
                else:
                    ti['seq_train'].extend(seqs_in_clip)
                    ti['n_seq_train'] += len(seqs_in_clip)
                    ti['n_frames_train'] += n_frames + (len(seqs_in_clip) - 1) * frame_shift

        ds_train.extend(ti['seq_train'])
        ds_test.extend(ti['seq_test'])


    # Calculate optimal controls
    inv_results_train = inv_results_test = None
    if data_type == 'xyz_inv':
        # todo
        raise RuntimeError('INV DATASET TODO!')

    # Write txt file
    breakdown = ''
    for tag_id, ti in tag_info.items():
        breakdown += f'\nClass: {LABELS[tag_id]}\n'
        breakdown += f'\tFrames available (theoretical): {ti["n_frames_all"]}\n'
        breakdown += f'\tTraining frames: {ti["n_frames_train"]} (target={ti["n_frames_target_train"]})\n'
        breakdown += f'\tTesting frames: {ti["n_frames_test"]} (target={ti["n_frames_target_test"]})\n'
        n = ti["n_frames_train"] + ti["n_frames_test"]
        if n > 0:
            breakdown += f'\tTest/train split: {ti["n_frames_train"] / n:.2f}\n'
        breakdown += f'\tTraining sequences: {ti["n_seq_train"]}\n'
        breakdown += f'\tTesting sequences: {ti["n_seq_test"]}\n'

    # Save dataset
    DS.set_trajectories(train=ds_train, test=ds_test)
    DS.save()

    return DS


def generate_or_load_data(
        data_type,
        rebuild_dataset=False,
        train_test_split=0.8,
        n_frames=10,
        frame_shift=5,
        n_cpca_components=2,
        restrict_classes=None,
        restrict_concs=None,
        restrict_length=None,
        include_mirrors=False,
        inv_opt_params={},
        printout=False,
):
    def cprint(x):
        if printout:
            print(x)

    cprint('---- Data ----')
    n_cpca_components = n_cpca_components if data_type == 'cpca' else ''
    ds_id = f'{data_type}{n_cpca_components}_f{n_frames}_s{frame_shift}'
    if restrict_classes is not None:
        ds_id += '_g' + ','.join([str(c) for c in restrict_classes])
    if restrict_concs is not None:
        ds_id += '_c' + ','.join([str(c) for c in restrict_concs])
    if restrict_length is not None:
        ds_id += '_l' + str(restrict_length)
    if include_mirrors:
        ds_id += '_m'
    ds_path = DATASETS_DIR + f'/{ds_id}'

    loaded = False
    if not rebuild_dataset:
        try:
            ds = TrajectoryDataset.objects.get(
                data_type=data_type,
                train_test_split_target=train_test_split,
                n_frames=n_frames,
                frame_shift=frame_shift,
                n_cpca_components=n_cpca_components,
                restrict_classes=restrict_classes,
                restrict_concs=restrict_concs,
                restrict_length=restrict_length,
                include_mirrors=include_mirrors,
                inv_opt_params=inv_opt_params,
            )
            loaded = True
        except DoesNotExist:
            print('Dataset could not be loaded from database.')

    if not loaded:
        print(f'Generating new dataset...')
        ds_all, ds_train, ds_test, meta_all, meta_train, meta_test, inv_results_train, inv_results_test = generate_dataset(
            ds_path,
            data_type,
            train_test_split=train_test_split,
            n_frames=n_frames,
            frame_shift=frame_shift,
            n_cpca_components=n_cpca_components,
            restrict_tags=restrict_classes,
            restrict_concs=restrict_concs,
            restrict_length=restrict_length,
            include_mirrors=include_mirrors,
            inv_opt_params=inv_opt_params,
        )

    return ds


def get_data_loaders(
        data_type,
        batch_size,
        rebuild_dataset=False,
        n_frames=10,
        frame_shift=5,
        n_cpca_components=2,
        restrict_classes=None,
        include_mirrors=False,
        inv_opt_params={},
        train_test_split=0.8,
        augment=False,
        n_workers=4,
):
    ds = generate_or_load_data(
        data_type,
        rebuild_dataset=rebuild_dataset,
        train_test_split=train_test_split,
        n_frames=n_frames,
        frame_shift=frame_shift,
        n_cpca_components=n_cpca_components,
        restrict_classes=restrict_classes,
        include_mirrors=include_mirrors,
        inv_opt_params=inv_opt_params
    )

    ds_args = {
        'ds_id': ds_id,
        'n_frames': n_frames,
        'restrict_classes': restrict_classes,
        'include_mirrors': include_mirrors,
        'augment': augment,
    }

    if data_type == 'xyz':
        cls = XYZDataset
    elif data_type == 'xyz_inv':
        cls = XYZInvDataset
    elif data_type == 'bishop':
        cls = BishopDataset
    elif 'cpca' in data_type:
        cls = CPCADataset
        ds_args['n_components'] = n_cpca_components
    else:
        raise Exception(f'Data type "{data_type}" not recognised')

    loaders = {}
    for tt in ['train', 'test']:
        # if 'inv' not in data[tt]:
        #     data[tt]['inv'] = None

        # dataset = cls(
        #     data=data[tt]['ds'],
        #     meta=data[tt]['meta'],
        #     inverse_data=data[tt]['inv'],
        #     **ds_args
        # )
        dataset = cls(
            data=data[tt]['ds'],
            meta=data[tt]['meta'],
            inverse_data=data[tt]['inv'],
            **ds_args
        )
        loaders[tt] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True
        )

    return loaders['train'], loaders['test']
