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



class DownStreamAudioDataset(Dataset):
    def __init__(self, root_dir, json_file, class_map_json, cfg, mode, clip_preprocess=None, root_dir_2=None, padding_mode='mirror'):
        self.modality = 'audio'
        self.root_dir = root_dir
        self.root_dir_2 = root_dir_2
        self.json_file = json_file
        self.class_map_json = class_map_json
        self.cfg = cfg
        self.mode = mode
        self.clip_preprocess = clip_preprocess  # Add CLIP preprocessing
        self.padding_mode = padding_mode

        # Load the class map
        self.classes = self._load_class_map()
        
        if len(self.classes) > 0:
            self.num_classes = len(self.classes)
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
        self.sub_root_dir_2 = []

        self.audio_files, self.audio_labels = self._load_json_file()
        
        # CLIP expects RGB images with 3 channels, so we'll adapt our spectrograms accordingly
        self.convert_to_rgb = True

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
                    continue

                audio_files.append(audio_path)
                # Convert label list to one-hot encoded vector if class map exists
                if isinstance(d['label'], list):
                    # Initialize an empty label tensor
                    label_tensor = torch.zeros(self.num_classes)

                    # For each label in the list, set the corresponding index to 1
                    for label_name in d['label']:
                        if label_name in self.classes.keys():
                            label_tensor[int(label_name)] = 1.0
                        audio_labels.append(label_tensor)
                else:
                    # Keep as is if no class map
                    audio_labels.append(d['label'])
                
        return audio_files, audio_labels

    def _wav2fbank(self, filename, filename2=None):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
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
                # print(f"Padding {filename} with {p} frames to reach target length {target_length}")
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


    def _spectrogram_to_clip_input(self, fbank):
        """
        Convert audio spectrogram to format expected by CLIP's ViT
        
        Args:
            fbank: Mel spectrogram of shape [freq_bins, time_frames]
            
        Returns:
            Tensor of shape [3, 224, 224] compatible with CLIP ViT
        """
        # Normalize to 0-1 range for conversion to image
        fbank_norm = (fbank - fbank.min()) / (fbank.max() - fbank.min() + 1e-6)
        
        # Convert to 3 channel image (repeat across channels)
        if self.convert_to_rgb:
            fbank_img = torch.repeat_interleave(fbank_norm.unsqueeze(0), 3, dim=0)
        else:
            # Use colormap for better visual representation
            # This is a more advanced conversion that might help CLIP better understand the audio
            fbank_np = fbank_norm.numpy()
            
            # Create RGB channels using different color mappings
            # Red channel - original spectrogram
            r_channel = fbank_np
            
            # Green channel - frequency gradient emphasis
            g_channel = np.zeros_like(fbank_np)
            for i in range(fbank_np.shape[0]):
                g_channel[i,:] = fbank_np[i,:] * (i / fbank_np.shape[0])
                
            # Blue channel - temporal gradient emphasis
            b_channel = np.zeros_like(fbank_np)
            for i in range(fbank_np.shape[1]):
                b_channel[:,i] = fbank_np[:,i] * (i / fbank_np.shape[1])
            
            # Combine channels
            fbank_img = torch.stack([
                torch.from_numpy(r_channel),
                torch.from_numpy(g_channel), 
                torch.from_numpy(b_channel)
            ], dim=0)
        
        # Resize to CLIP's expected input size (typically 224x224)
        if self.clip_preprocess:
            # Use CLIP's own preprocessing
            return self.clip_preprocess(fbank_img.unsqueeze(0)).squeeze(0)
        else:
            # Fallback to manual resize
            resized = torch.nn.functional.interpolate(
                fbank_img.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Apply normalization expected by CLIP
            # CLIP uses the same normalization as ImageNet
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            normalized = (resized - mean) / std
            
            return normalized

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        Returns audio spectrogram in a format compatible with CLIP's ViT model
        """
        # Get standard spectrogram processing (mostly keep existing code)
        if random.random() < self.mixup and self.mode == 'train':
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
        
        # SpecAug, not do for eval set
        if self.mode == 'train':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            
            fbank = torch.transpose(fbank, 0, 1)
            
            # this is just to satisfy new torchaudio version
            fbank = fbank.unsqueeze(0)
            
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
        
        if self.noise and self.mode == 'train':
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # Convert to format expected by CLIP
        clip_input = self._spectrogram_to_clip_input(fbank)
        
        return clip_input, label, self.modality


