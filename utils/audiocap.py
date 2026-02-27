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



class AudioCapDataset(Dataset):

    def __init__(self, root_dir, json_file, cfg, mode, audio_feat_dir=None, caption_feature_dir=None, padding_mode='mirror', clip_align=False, audio_augmentation=True, text_seq_len=256, extract_features=False):
        self.modality = 'audio'
        self.root_dir = root_dir
        self.json_file = json_file
        self.cfg = cfg
        self.mode = mode
        self.padding_mode = padding_mode  # 'zero_pad' or 'mirror'
        self.clip_align = clip_align
        self.audio_augmentation = audio_augmentation
        self.text_seq_len = text_seq_len
        self.extract_features = extract_features
        self.audio_feat_dir = audio_feat_dir
        self.caption_feature_dir = caption_feature_dir
        


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

        self.audio_files, self.audio_labels = self._load_json_file()

        # if self.clip_align and self.extract_features:
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        if self.clip_align and not self.extract_features:
            # filter all samples without feature extracted
            filtered_data = []
            for sample in self.audio_files:
                audio_feature_path = os.path.join(self.audio_feat_dir, sample.split('/')[-1].split('.')[0] + '.npy')
                caption_feature_path = os.path.join(self.caption_feature_dir, sample.split('/')[-1].split('.')[0] + '.npy')
                if os.path.exists(audio_feature_path) and os.path.exists(caption_feature_path):
                    filtered_data.append(sample)

            print(f'Filtered {len(self.audio_files) - len(filtered_data)} samples without extracted features.')
            self.audio_files = filtered_data




    def _load_json_file(self):
        print(f"Loading audio files and captions from {self.json_file}...")
        audio_files = []
        audio_labels = []
        temporal_idx  = []
        num_wav_not_exist = 0
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                audio_path = os.path.join(self.root_dir, d['name'])

                audio_path = audio_path.split('.')[0] + '.wav'
                if not os.path.exists(audio_path):
                    num_wav_not_exist += 1
                    continue

                audio_files.append(audio_path)
                audio_labels.append(d['labels'])

        print(f"Loaded {len(audio_files)} audio files from {self.json_file}, skipped {num_wav_not_exist} missing files.")
                
        return audio_files, audio_labels

    def _wav2fbank(self, filename):
        # no mixup
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


        return fbank, 0


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
        datum = self.audio_files[index]
        caption = self.audio_labels[index]
        

        
        if self.clip_align and self.extract_features:
            # For CLIP-style alignment training
            # Apply one random augmentation to the spectrogram
            fbank, mix_lambda = self._wav2fbank(datum)
            fbank = self.apply_spectrogram_augmentation(fbank)
            # Tokenize caption
            encoded = self.tokenizer.encode_plus(
                caption,
                add_special_tokens=True,
                max_length=self.text_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_tokens = encoded['input_ids'].squeeze(0)
            return fbank, caption_tokens, self.modality, datum.split('/')[-1].split('.')[0]
        
        elif self.clip_align and not self.extract_features:
            # read the extreacted feature
            audio_feature_path = os.path.join(self.audio_feat_dir, datum.split('/')[-1].split('.')[0] + '.npy')
            caption_feature_path = os.path.join(self.caption_feature_dir, datum.split('/')[-1].split('.')[0] + '.npy')
            audio_feature = torch.from_numpy(np.load(audio_feature_path))
            caption_feature = torch.from_numpy(np.load(caption_feature_path))
            return audio_feature, caption_feature, self.modality

        else:
            # for zero-shot eval
            fbank, mix_lambda = self._wav2fbank(datum)
            fbank = self.apply_spectrogram_augmentation(fbank)
            # Tokenize caption
            encoded = self.tokenizer.encode_plus(
                caption,
                add_special_tokens=True,
                max_length=self.text_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_tokens = encoded['input_ids'].squeeze(0)
            return fbank, caption_tokens, self.modality, index


            # fbank, mix_lambda = self._wav2fbank(datum)
            # # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            # fbank = self.apply_spectrogram_augmentation(fbank)
            # return fbank, label, self.modality


        
