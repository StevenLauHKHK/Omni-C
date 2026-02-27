import os
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt



class AudioDataset(Dataset):

    def __init__(self, root_dir, json_file, class_map_json, cfg, mode, root_dir_2=None, padding_mode='mirror', return_double_augmentations=False, clip_align=False, audio_augmentation=True):
        self.modality = 'audio'
        self.root_dir = root_dir
        self.root_dir_2 = root_dir_2
        self.json_file = json_file
        self.class_map_json = class_map_json
        self.cfg = cfg
        self.mode = mode
        self.padding_mode = padding_mode  # 'zero_pad' or 'mirror'
        self.return_double_augmentations = return_double_augmentations
        self.clip_align = clip_align
        self.audio_augmentation = audio_augmentation

        # Load the class map
        self.class_map = self._load_class_map()
        if len(self.class_map)>0:
            self.num_classes = len(self.class_map)
        else:
            self.num_classes = cfg.DATA.NUM_CLASSES  

        # Set the audio configuration
        self.melbins = cfg.AUDIO.MELBINS  # 128
        self.freqm = cfg.AUDIO.FREQM  # 0
        self.timem = cfg.AUDIO.TIMEM # 0
        self.target_length = cfg.AUDIO.TARGET_LENGTH # 1024
        self.mixup = cfg.AUG.MIXUP  # 0
        self.norm_mean = cfg.AUDIO.NORM_MEAN # -4.2677393
        self.norm_std = cfg.AUDIO.NORM_STD  # 4.5689974
        self.skip_norm = cfg.AUDIO.SKIP_NORM # False
        # if add noise for data augmentation
        self.noise = cfg.AUDIO.NOISE # False
        self.sub_root_dir_2 = ['train-other-500', 'train-clean-100', 'train-clean-360']

        self.audio_files, self.audio_labels = self._load_json_file()

    def _load_class_map(self):
        """
        Load the class map from JSON file
        Returns: Dictionary mapping class names to indices
        """
        if self.class_map_json is None:
            print("Warning: No class map JSON file provided")
            return {}
            
        if not os.path.exists(self.class_map_json):
            print(f"Warning: Class map file {self.class_map_json} not found")
            return {}
        
        with open(self.class_map_json, 'r') as f:
            class_map = json.load(f)
        
        print(f"Loaded {len(class_map)} classes from class map")
        return class_map


    def _load_json_file(self):
        audio_files = []
        audio_labels = []
        temporal_idx  = []
        num_wav_not_exist = 0
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                if d['name'].split('/')[0] in self.sub_root_dir_2 and self.root_dir_2 is not None:
                    audio_path = os.path.join(self.root_dir_2, d['name'])
                else:
                    audio_path = os.path.join(self.root_dir, d['name'])

                audio_path = audio_path.split('.')[0] + '.wav'
                
                if not os.path.exists(audio_path):
                    num_wav_not_exist += 1
                    continue

                audio_files.append(audio_path)
                # Convert label list to one-hot encoded vector if class map exists
                if self.class_map and isinstance(d['label'], list):
                    # Initialize an empty label tensor
                    label_tensor = torch.zeros(self.num_classes)

                    # For each label in the list, set the corresponding index to 1
                    for label_name in d['label']:
                        if label_name in self.class_map.keys():
                            label_tensor[int(label_name)] = 1.0
                    audio_labels.append(label_tensor)
                else:
                    # Keep as is if no class map
                    audio_labels.append(d['label'])

        print(f"Loaded {len(audio_files)} audio files from {self.json_file}, skipped {num_wav_not_exist} missing files.")
                
        return audio_files, audio_labels

    def _wav2fbank(self, filename, filename2=None):
        # no mixup
        if filename2 == None:
            try:
                waveform, sr = torchaudio.load(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                # Return a zero tensor if loading fails
                dummy_waveform = torch.zeros(1, 32000)  # 1 second of silence at 32kHz
                sr = 32000
                waveform = dummy_waveform

            # print(f"Loaded {filename} with sample rate {sr}")
            if sr != 32000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
                waveform = resampler(waveform)
                sr = 32000  # Set the sample rate to 32000 Hz
            # Check if waveform is too short and pad if necessary
            min_waveform_length = 800  # Minimum length needed for window size
            if waveform.shape[1] < min_waveform_length:
                padding = torch.zeros(1, min_waveform_length)
                padding[0, :waveform.shape[1]] = waveform
                waveform = padding
                print(f"Warning: {filename} was too short ({waveform.shape[1]} samples), padded to {min_waveform_length}")
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=39.12)
        # print(f"Extracted fbank from {filename} with shape {fbank.shape}")

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            if self.padding_mode == 'zero_pad':
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)

            else:
                # Create a new tensor with the target length
                padded_fbank = torch.zeros(target_length, fbank.shape[1])

                # Copy the original content
                padded_fbank[:n_frames, :] = fbank

                # Mirror padding approach
                frames_to_mirror = min(n_frames, p)  # Number of frames to mirror
                mirror_frames = torch.flip(fbank[-frames_to_mirror:, :], [0])  # Flip the last frames

                # Fill with mirrored frames
                for i in range(min(p, frames_to_mirror)):
                    padded_fbank[n_frames + i, :] = mirror_frames[i % frames_to_mirror, :]

                # If we need more padding than we have frames to mirror, repeat the pattern
                if p > frames_to_mirror:
                    remaining = p - frames_to_mirror
                    for i in range(remaining):
                        # Repeat the mirror pattern
                        padded_fbank[n_frames + frames_to_mirror + i, :] = padded_fbank[n_frames + (i % frames_to_mirror), :]
                
                # Replace the original fbank with the padded version
                fbank = padded_fbank
                # print(f"Mirror-padding {filename} with {p} frames to reach target length {target_length}")
                
                # Save padded spectrogram
                # if random.random() < 0.01:  # Only save ~1% of samples
                #     plt.figure(figsize=(10, 4))
                #     plt.imshow(fbank.numpy().T, aspect='auto', origin='lower')
                #     plt.colorbar(format='%+2.0f dB')
                #     plt.title(f'Padded Spectrogram - Added {p} frames')
                #     plt.xlabel('Time Frames')
                #     plt.ylabel('Mel Filterbank')
                #     plt.tight_layout()
                #     save_dir = 'padded_spectrograms'
                #     base_filename = os.path.split(filename)[1].split('.')[0]
                #     plt.savefig(os.path.join(save_dir, f'{base_filename}_padded.png'))
                #     plt.close()
        
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def crop_and_augment(self, fbank_input):
        """Crop audio into two segments and apply augmentation with mirror padding"""
        total_frames = fbank_input.shape[0]  # 256
        segment_length = total_frames // 2   # 128
        
        # Random starting points for more variety
        start_1 = random.randint(0, total_frames - segment_length)
        start_2 = random.randint(0, total_frames - segment_length)
        
        # Ensure segments don't overlap too much
        while abs(start_1 - start_2) < segment_length // 2:
            start_2 = random.randint(0, total_frames - segment_length)
        
        segment_1 = fbank_input[start_1:start_1 + segment_length, :]
        segment_2 = fbank_input[start_2:start_2 + segment_length, :]
        
        # Apply mirror padding to restore target_length
        segment_1 = self.apply_mirror_padding(segment_1)
        segment_2 = self.apply_mirror_padding(segment_2)
        
        # Apply freq/time masking (no shift needed)
        seg_1 = self.apply_spectrogram_augmentation(segment_1, time_shift=False)
        seg_2 = self.apply_spectrogram_augmentation(segment_2, time_shift=False)
        
        return seg_1, seg_2

    def apply_mirror_padding(self, fbank_segment):
        """Apply mirror padding to restore target_length"""
        current_length = fbank_segment.shape[0]
        target_length = self.target_length
        
        if current_length >= target_length:
            # If segment is already long enough, just truncate
            return fbank_segment[:target_length, :]
        
        # Need to pad
        p = target_length - current_length
        
        # Create a new tensor with the target length
        padded_fbank = torch.zeros(target_length, fbank_segment.shape[1])
        
        # Copy the original content
        padded_fbank[:current_length, :] = fbank_segment
        
        # Mirror padding approach (same as your _wav2fbank method)
        frames_to_mirror = min(current_length, p)  # Number of frames to mirror
        mirror_frames = torch.flip(fbank_segment[-frames_to_mirror:, :], [0])  # Flip the last frames
        
        # Fill with mirrored frames
        for i in range(min(p, frames_to_mirror)):
            padded_fbank[current_length + i, :] = mirror_frames[i % frames_to_mirror, :]
        
        # If we need more padding than we have frames to mirror, repeat the pattern
        if p > frames_to_mirror:
            remaining = p - frames_to_mirror
            for i in range(remaining):
                # Repeat the mirror pattern
                padded_fbank[current_length + frames_to_mirror + i, :] = padded_fbank[current_length + (i % frames_to_mirror), :]
        
        return padded_fbank

    def apply_time_shift_on_spectrogram(self, fbank, max_shift):
        """Apply circular time shift on the spectrogram"""
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(fbank, shifts=shift, dims=0)

    def apply_spectrogram_augmentation(self, fbank_input, time_shift=True):
        """Apply spectrogram augmentation to the input fbank"""
        # Apply circular time shift FIRST (before other augmentations)
        if self.return_double_augmentations and time_shift and self.audio_augmentation:
            # For SimCLR, apply random circular time shift
            max_shift = int(0.1 * fbank_input.shape[0])  # 10% of total frames
            fbank_input = self.apply_time_shift_on_spectrogram(fbank_input, max_shift)
        
        # SpecAug, not do for eval set
        if self.mode == 'train' and self.audio_augmentation:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)

        fbank = torch.transpose(fbank_input, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        
        # SpecAug, not do for eval set
        if self.mode == 'train' and self.audio_augmentation:
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)

        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        else:
            pass

        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # Transpose from [time_frame_num, frequency_bins] to [frequency_bins, time_frame_num]
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)

        return fbank


    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.audio_files[index]
            label_1 = self.audio_labels[index]
            mix_sample_idx = random.randint(0, len(self.audio_files)-1)
            mix_datum = self.audio_files[mix_sample_idx]
            label_2 = self.audio_labels[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum, mix_datum)
            # initialize the label
            label = torch.max(label_1, label_2)
            
        else:
            datum = self.audio_files[index]
            label = self.audio_labels[index]
            fbank, mix_lambda = self._wav2fbank(datum)

        if self.return_double_augmentations:
            # Apply two different random augmentations to the same spectrogram
            if self.cfg.AUDIO.PRETRAIN_CROP_AND_AUGMENT:
                fbank_1, fbank_2 = self.crop_and_augment(fbank)
            else:
                fbank_1 = self.apply_spectrogram_augmentation(fbank)
                fbank_2 = self.apply_spectrogram_augmentation(fbank)

            # Return both augmented versions
            return fbank_1, fbank_2, label, self.modality
        
        elif self.clip_align and self.class_map:
            # Apply one random augmentation to the spectrogram
            fbank = self.apply_spectrogram_augmentation(fbank)
            
            if torch.is_tensor(label) and label.dim() > 0 and label.sum() > 0:
                # Random pick a class name from the multi-labels for CLIP alignment
                label_indices = (label == 1).nonzero(as_tuple=True)[0]
                if len(label_indices) > 0:
                    indices_list = label_indices.tolist()
                    random_index = random.choice(indices_list)
                    label_name = self.class_map[str(random_index)]
                    return fbank, label_name, self.modality
                else:
                    return fbank, "unknown", self.modality
            else:
                label_name = self.class_map[str(label)]
                return fbank, label_name, self.modality

        else:
            # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            fbank = self.apply_spectrogram_augmentation(fbank)
            return fbank, label, self.modality

        
