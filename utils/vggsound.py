import os
import json
from PIL import Image
import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from .spec_augment import combined_transforms

class VGGSoundDataset(Dataset):
    def __init__(self, root_dir, json_file, cfg, mode, augmentation=False):
        self.modality = 'audio'
        self.root_dir = root_dir
        self.json_file = json_file
        self.cfg = cfg
        self.mode = mode
        self.augmentation = augmentation
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["val"]:
            self._num_clips = cfg.AUDIO.NUM_ENSEMBLE_VIEWS
        elif self.mode in ["test"]:
            self._num_clips = cfg.AUDIO.NUM_ENSEMBLE_VIEWS

        self.audio_paths, self.labels, self.temporal_idx = self._load_json_file()
        
    def _get_start_end_idx(self, audio_size, clip_size, clip_idx, num_clips):
        """
        Sample a clip of size clip_size from an audio of size audio_size and
        return the indices of the first and last sample of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the audio to
        num_clips clips, and select the start and end index of clip_idx-th audio
        clip.
        Args:
            audio_size (int): number of overall samples.
            clip_size (int): size of the clip to sample from the samples.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the audio to num_clips
                clips, and select the start and end index of the clip_idx-th audio
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given audio for testing.
        Returns:
            start_idx (int): the start sample index.
            end_idx (int): the end sample index.
        """
        delta = max(audio_size - clip_size, 0)
        if clip_idx == -1:
            # Random temporal sampling.
            start_idx = random.uniform(0, delta)
        else:
            # Uniformly sample the clip with the given index.
            start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def _extract_sound_feature(self, cfg, samples, start_idx, end_idx):
        if samples.shape[1] < int(round(cfg.AUDIO.SAMPLING_RATE * cfg.AUDIO.CLIP_SECS)):
            spectrogram = self._log_specgram(cfg, samples,
                                        window_size=cfg.AUDIO.WINDOW_LENGTH,
                                        step_size=cfg.AUDIO.HOP_LENGTH
                                        )
            num_timesteps_to_pad = cfg.AUDIO.NUM_FRAMES - spectrogram.shape[1]
            pad_tuple = (0, 0, num_timesteps_to_pad, 0)
            spectrogram = F.pad(spectrogram, pad_tuple)
        else:
            samples = samples[:, start_idx:end_idx]
            spectrogram = self._log_specgram(cfg, samples,
                                        window_size=cfg.AUDIO.WINDOW_LENGTH,
                                        step_size=cfg.AUDIO.HOP_LENGTH
                                        )
            if spectrogram.shape[1] < cfg.AUDIO.NUM_FRAMES:
                num_timesteps_to_pad = cfg.AUDIO.NUM_FRAMES - spectrogram.shape[1]
                pad_tuple = (0, 0, num_timesteps_to_pad, 0)
                spectrogram = F.pad(spectrogram, pad_tuple)

        return spectrogram

    def _log_specgram(self, cfg, audio, window_size=10,
                 step_size=5, eps=1e-6):
        nperseg = int(round(window_size * cfg.AUDIO.SAMPLING_RATE / 1e3))
        noverlap = int(round(step_size * cfg.AUDIO.SAMPLING_RATE / 1e3))

        window = torch.hann_window(nperseg)
        spec = torch.stft(audio, n_fft=2048, hop_length=noverlap, win_length=nperseg, pad_mode='constant', window=window, return_complex=True)
        mel_basis = torchaudio.transforms.MelScale(sample_rate=cfg.AUDIO.SAMPLING_RATE, n_mels=128)
        spec = abs(spec)
        mel_spec = torch.squeeze(mel_basis(spec)).numpy()
        log_mel_spec = np.log(mel_spec + eps)
        log_mel_spec = torch.tensor(log_mel_spec).unsqueeze(0)

        return log_mel_spec.permute(0, 2, 1)

    def _load_json_file(self):
        audio_files = []
        audio_labels = []
        temporal_idx  = []
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                audio_path = os.path.join(self.root_dir, d['name'].split('.mp4')[0] + '.wav')
                if not os.path.exists(audio_path):
                    continue
                for idx in range(self._num_clips):
                    audio_files.append(d['name'])
                    audio_labels.append(d['label'])
                    temporal_idx.append(idx)
        return audio_files, audio_labels, temporal_idx

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test", "val"]:
            temporal_sample_index = self.temporal_idx[idx]
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        audio_path = os.path.join(self.root_dir, self.audio_paths[idx].split('.mp4')[0] + '.wav')
        samples, sr = torchaudio.load(audio_path, normalization=True)
        
        start_idx, end_idx = self._get_start_end_idx(
            samples.shape[1],
            int(round(self.cfg.AUDIO.SAMPLING_RATE * self.cfg.AUDIO.CLIP_SECS)),
            temporal_sample_index,
            self.cfg.AUDIO.NUM_ENSEMBLE_VIEWS
        )
        

        spectrogram = self._extract_sound_feature(self.cfg, samples, int(start_idx), int(end_idx))

        if self.mode == "train" and self.augmentation:
            # Data augmentation.
            spectrogram_1 = spectrogram.clone()
            spectrogram_2 = spectrogram.clone()
            # C T F -> C F T
            spectrogram_1 = spectrogram_1.permute(0, 2, 1)
            spectrogram_2 = spectrogram_2.permute(0, 2, 1)
            # SpecAugment
            spectrogram_1 = combined_transforms(spectrogram_1)
            spectrogram_2 = combined_transforms(spectrogram_2)
            # C F T -> C T F
            spectrogram_1 = spectrogram_1.permute(0, 2, 1)
            spectrogram_2 = spectrogram_2.permute(0, 2, 1)
        else:
            spectrogram_1 = spectrogram
            spectrogram_2 = None

        label = self.labels[idx]
        
        if self.mode in ["train"]:
            return spectrogram_1, spectrogram_2, label, self.modality
        else:
            return spectrogram_1, spectrogram_2, label, idx, self.modality