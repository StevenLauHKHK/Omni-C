import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, root_dir, json_file, class_map_json, vocab_file, cfg, mode, text_seq_len=128, augment=False, 
                 p_mask=0.15, p_delete=0.1, mask_token_id=103, pad_token_id=0, 
                 return_double_augmentations=False):
        """
        Text Dataset Loader
        Args:
            root_dir (str): Root directory containing token files.
            json_file (str): JSON file with token file paths and labels.
            vocab_file (str): JSON file with vocabulary mapping.
            text_seq_len (int): Maximum sequence length for text.
            augment (bool): Whether to apply text augmentations.
            p_mask (float): Probability of masking a token.
            p_delete (float): Probability of deleting a token.
            mask_token_id (int): Token ID used for masking.
        """
        self.modality = 'text'
        self.root_dir = root_dir
        self.json_file = json_file
        self.class_map_json = class_map_json
        self.cfg = cfg
        self.mode = mode
        self.vocab_file = vocab_file
        self.text_seq_len = text_seq_len
        self.augment = augment
        self.p_mask = p_mask
        self.p_delete = p_delete
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.return_double_augmentations = return_double_augmentations

        # Load the class map
        self.class_map = self._load_class_map()
        if len(self.class_map)>0:
            self.num_classes = len(self.class_map)
        else:
            self.num_classes = cfg.DATA.NUM_CLASSES  

        # Load token paths and labels
        self.token_paths, self.labels = self._load_json_file()

        # Load vocabulary
        self.vocab = self._load_vocab()

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

    def _load_vocab(self):
        """Load vocabulary from a JSON file."""
        vocab = {}
        try:
            with open(self.vocab_file, 'r') as json_file:
                json_data = json.load(json_file)
                for word, token in json_data.items():
                    vocab[token] = word
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")
        return vocab

    def augment_text(self, tokens):
        """
        Apply augmentations to a single token sequence for contrastive learning.
        Args:
            tokens (list[int] or np.array): Input token sequence.
        Returns:
            list[int]: Augmented token sequence.
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        augmented_seq = []
        for token in tokens:
            # Skip padding tokens from augmentation
            if token == self.pad_token_id:
                augmented_seq.append(token)
                continue

            prob = random.random()
            if prob < self.p_mask:
                augmented_seq.append(self.mask_token_id)  # Mask the token
            elif prob < self.p_mask + self.p_delete:
                continue  # Delete the token
            else:
                augmented_seq.append(token)  # Keep the token
        return augmented_seq


    def apply_stronger_augmentations(self, tokens):
        """
        Apply stronger augmentations for better contrastive learning.
        Args:
            tokens (list[int] or np.array): Input token sequence.
        Returns:
            list[int]: Strongly augmented token sequence.
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        # Remove padding for augmentation
        non_pad_tokens = [token for token in tokens if token != self.pad_token_id]

        augmented_seq = []

        # Apply multiple augmentation strategies
        for i, token in enumerate(non_pad_tokens):
            prob = random.random()

            if prob < self.p_mask * 1.5:  # Increased masking probability
                augmented_seq.append(self.mask_token_id)  # Mask the token
            elif prob < self.p_mask * 1.5 + self.p_delete * 1.2:  # Increased deletion probability
                continue  # Delete token
            elif prob < self.p_mask * 1.5 + self.p_delete * 1.2 + 0.05:  # Token shuffling
                # Simple local shuffling: swap with next token occasionally
                if i < len(non_pad_tokens) - 1 and random.random() < 0.5:
                    augmented_seq.extend([non_pad_tokens[i+1], token])
                    non_pad_tokens[i+1] = None  # Mark as processed
                else:
                    augmented_seq.append(token)
            elif non_pad_tokens[i] is not None:  # Make sure we don't process swapped tokens twice
                augmented_seq.append(token)  # Keep the token
        
        # Clean up None values from shuffling
        augmented_seq = [token for token in augmented_seq if token is not None]
        return augmented_seq

    def pad_sequence(self, tokens):
        """
        Pad or truncate token sequence to desired length.
        Args:
            tokens (list[int]): Input token sequence.
        Returns:
            np.array: Padded token sequence.
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if len(tokens) > self.text_seq_len:
            tokens = tokens[:self.text_seq_len]
        else:
            tokens = tokens + [self.pad_token_id] * (self.text_seq_len - len(tokens))
        
        return np.array(tokens)

    def _load_json_file(self):
        """Load token file paths and labels from a JSON file."""
        token_files = []
        labels = []
        try:
            with open(self.json_file, 'r') as json_file:
                json_data = json.load(json_file)
                for d in json_data:
                    token_files.append(d['name'])
                    labels.append(d['label'])
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.json_file}")
        return token_files, labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.token_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Args:
            idx (int): Index of the sample.
        Returns:
            For contrastive learning (return_double_augmentations=True):
                tuple: ((token_seq_1, token_seq_2), label, modality)
            For standard training (return_double_augmentations=False):
                tuple: (token_seq, label, modality)
        """
        token_path = os.path.join(self.root_dir, self.token_paths[idx])
        try:
            tokens = np.load(token_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Token file not found: {token_path}")

        label = self.labels[idx]

        if self.return_double_augmentations:
            # For contrastive learning: return two different augmented versions
            if self.augment:
                # Apply different augmentation strategies for diversity
                tokens_1 = self.augment_text(tokens.copy())
                tokens_2 = self.apply_stronger_augmentations(tokens.copy())
            else:
                # Even without augment=True, we can apply minimal augmentations for contrastive learning
                tokens_1 = self.augment_text(tokens.copy())
                tokens_2 = self.augment_text(tokens.copy())
        
            # Pad both sequences
            tokens_1 = self.pad_sequence(tokens_1)
            tokens_2 = self.pad_sequence(tokens_2)

            return torch.tensor(tokens_1), torch.tensor(tokens_2), label, self.modality

        else:
            # For standard training: return single (possibly augmented) version
            if self.augment:
                tokens = self.augment_text(tokens)

            # Pad sequence
            tokens = self.pad_sequence(tokens)
            return torch.tensor(tokens), label, self.modality