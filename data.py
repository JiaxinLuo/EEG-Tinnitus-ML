import matplotlib.pyplot as plt
import random
import gc
import traceback
import scipy
import numpy as np

from pathlib import Path
from pdb import set_trace as st

import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio


class EEGSubset(torch.utils.data.Dataset):

    def __init__(self):
        dataset_root = Path(
            '/mnt/lustre/OneDrive/sensetime-server-mirror/EEG-Data/Preprocessed-upload'
        ) / 'Stand'
        tinn_samples = ['AS_tinn_500and5kHz_mcattn-2_sess1_set1_ICAREV.mat']
        cntl_samples = ['BH_cntl_2-Stream_mcattn-2_sess1_set1_ICAREV.mat']
        self.waves = []

        for sample_name in tinn_samples:
            sample_mat = scipy.io.loadmat(str(dataset_root / sample_name))
            allepochs = sample_mat['allepochs'][0, 0].astype(
                np.float32)  # TODO 360, 358, 64
            allepochs = allepochs.transpose(0, 2, 1)  # 360, 64, 358
            playvecs = sample_mat['playvecs'][0, 0]  # 1, 360
            # st()
            for i in range(allepochs.shape[0]):
                self.waves.append((allepochs[i], playvecs[0, i], 'tinn', 1))

        for sample_name in cntl_samples:
            sample_mat = scipy.io.loadmat(str(dataset_root / sample_name))
            allepochs = sample_mat['allepochs'][0,
                                                0].astype(np.float32)  # TODO
            allepochs = allepochs.transpose(0, 2, 1)  # 360, 64, 358
            playvecs = sample_mat['playvecs'][0, 0]
            for i in range(allepochs.shape[0]):
                self.waves.append((allepochs[i], playvecs[0, i], 'cntl', 0))

        # todo, preprocess data into independent dataset. high-low-passive; names; 360/540 files.
        # here, just load into the memory for testing
        gc.collect()

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx]


class EEGFull(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_root:
        str = '/mnt/lustre/OneDrive/sensetime-server-mirror/EEG-Data/Preprocessed-upload',
            subset='Stand',
            training=True):

        subset_index = (0, 0)
        # subset_index = (0, 1)
        # subset_index = (1,0)
        # subset_index = (1,1)

        print('using index', subset_index)

        self.waves = []
        all_sample_paths = sorted((Path(dataset_root) / subset).glob('*.mat'))
        if training:
            # all_sample_paths = all_sample_paths[:-5]
            all_sample_paths = all_sample_paths[:10]
        else:
            all_sample_paths = all_sample_paths[-5:]  # hold out
            # all_sample_paths = all_sample_paths[5:7]  # hold out
            # all_sample_paths = all_sample_paths[5:10]  # hold out
            # all_sample_paths = all_sample_paths[5:15]  # hold out

        for sample_path in all_sample_paths:
            if 'tinn' in sample_path.stem:
                label = ('tinn', 1)
            else:
                assert 'cntl' in sample_path.stem
                label = ('cntl', 0)
            try:
                sample_mat = scipy.io.loadmat(str(sample_path))
                allepochs = sample_mat['allepochs'][subset_index].astype(
                    np.float32)  # TODO 360, 358, 64
                allepochs = allepochs.transpose(0, 2, 1)  # 360, 64, 358
                playvecs = sample_mat['playvecs'][subset_index]  # 1, 360

                for i in range(allepochs.shape[0]):
                    self.waves.append((allepochs[i], playvecs[0, i], *label))
                del playvecs, allepochs
            except:
                traceback.print_exc()
                st()

        gc.collect()

        # todo, preprocess data into independent dataset. high-low-passive; names; 360/540 files.
        # here, just load into the memory for testing
    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx]


class EEG_Standard(torch.utils.data.Dataset):
    """just process the standard dataset, 4 types of return value:

    blk 0/1 (HF/LF) * attend 3/4

    set "tinn" "cntl" to differentiate which group to return

    flags: 0 for cntl, and 1 for tinn

    ? how to split train and test?
    """

    def __init__(
            self,
            dataset_root: str = '/mnt/lustre/2023/processed_eeg_data/standard/',
            # subset='Stand',
            blk_idx=0,  # 0 for HF and 1 for LF
            tone_idx=3,  # 3 or 4
            trian_split_ratio=0.8, # how many samples to use for training
            training=True):
        # denote tinn and cntl (label 1 and 0)? return all.

        if blk_idx == 0:
            blk_name = 'Att-HF'
        else:
            blk_name = 'Att-LF'

        # assert tone_idx in [3, 4]
        assert tone_idx in [1,2,3, 4]

        dataset_path = Path(dataset_root)
        pos_ids = dataset_path.glob('*_tinn')

        neg_idx = dataset_path.glob('*_cntl')

        def load_blk_tone(id_paths):
            # return the path of the mat file, regarding the blk and tone
            files = []
            for id_path in id_paths:
                files.extend(
                    list((id_path / blk_name / f'{tone_idx}').glob('*.npy')))
            return files

        self.pos_file_paths = load_blk_tone(pos_ids)
        self.neg_file_paths = load_blk_tone(neg_idx)

        # print('pos', len(self.pos_file_paths))
        # print('neg', len(self.neg_file_paths))

        self.file_paths = [[fp, 0] for fp in self.neg_file_paths
                           ] + [[fp, 1] for fp in self.pos_file_paths]

        random.shuffle(self.file_paths) # seed

        if training:
            self.file_paths = self.file_paths[:int(len(self.file_paths) * trian_split_ratio)]
            print('pos in training', len([fp for fp in self.file_paths if fp[1] == 1]))
        else:
            self.file_paths = self.file_paths[int(len(self.file_paths) * trian_split_ratio):]
            print('pos in test', len([fp for fp in self.file_paths if fp[1] == 1]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx]
        # (358, 64)
        waveform = np.load(file_path).astype(np.float32)  # ? any normalizations, no ideas
        return waveform, label