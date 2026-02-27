import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class WiKiDataset(Dataset):
    def __init__(self, root_dir, json_file, vocab_file, token_meta_json_file, token_pairs_json_file, 
                 text_seq_len=128, augment=False, p_mask=0.15, p_delete=0.1, mask_token_id=103, 
                 pad_token_id=0, enable_nsp=False, return_double_augmentations=False, contrastive_mode='simcse'):
        """
        Wikipedia Dataset Loader for Contrastive Learning
        Args:
            root_dir (str): Root directory containing token files.
            json_file (str): JSON file with token file paths and labels.
            vocab_file (str): JSON file with vocabulary mapping.
            token_meta_json_file (str): JSON file with token metadata.
            token_pairs_json_file (str): JSON file with positive pairs for NSP.
            text_seq_len (int): Maximum sequence length for text.
            augment (bool): Whether to apply text augmentations.
            p_mask (float): Probability of masking a token.
            p_delete (float): Probability of deleting a token.
            mask_token_id (int): Token ID used for masking.
            pad_token_id (int): Token ID used for padding.
            enable_nsp (bool): Enable Next Sentence Prediction mode.
            return_double_augmentations (bool): If True, return two augmented versions for contrastive learning.
            contrastive_mode (str): Type of contrastive learning ('simcse', 'semantic', 'discourse').
        """
        self.modality = 'text'
        self.enable_nsp = enable_nsp
        self.root_dir = root_dir
        self.json_file = json_file
        self.vocab_file = vocab_file
        self.token_meta_json_file = token_meta_json_file
        self.token_pairs_json_file = token_pairs_json_file
        self.text_seq_len = text_seq_len
        self.augment = augment
        self.p_mask = p_mask
        self.p_delete = p_delete
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.return_double_augmentations = return_double_augmentations
        self.contrastive_mode = contrastive_mode

        # Load token paths and labels
        self.token_paths, self.labels = self._load_json_file()

        # Load vocabulary
        self.vocab = self._load_vocab()

        # Load document pairs for NSP if enabled
        print(f"NSP enabled: {self.enable_nsp}")
        print(f"Contrastive mode: {self.contrastive_mode}")
        print(f"Return double augmentations: {self.return_double_augmentations}")
        
        if self.enable_nsp or self.contrastive_mode in ['semantic', 'discourse']:
            self.positive_base_path_1, self.positive_base_path_2, self.positive_chunk_num_1, self.positive_chunk_num_2 = self._load_json_positive_pairs()

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
        Apply light augmentations to a single token sequence (SimCSE-style).
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
        Apply stronger augmentations for better contrastive diversity.
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
            
            if prob < self.p_mask * 1.5:  # Increased masking
                augmented_seq.append(self.mask_token_id)
            elif prob < self.p_mask * 1.5 + self.p_delete * 1.2:  # Increased deletion
                continue
            elif prob < self.p_mask * 1.5 + self.p_delete * 1.2 + 0.05:  # Token shuffling
                # Simple local shuffling: swap with next token occasionally
                if i < len(non_pad_tokens) - 1 and random.random() < 0.5:
                    augmented_seq.extend([non_pad_tokens[i+1], token])
                    if i + 1 < len(non_pad_tokens):
                        non_pad_tokens[i+1] = None  # Mark as processed
                else:
                    augmented_seq.append(token)
            else:
                augmented_seq.append(token)
        
        # Clean up None values from shuffling
        augmented_seq = [token for token in augmented_seq if token is not None]
        
        return augmented_seq

    def pad_sequence(self, tokens):
        """
        Pad or truncate token sequence to desired length.
        Args:
            tokens (list[int] or np.array): Input token sequence.
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

    def _load_json_positive_pairs(self):
        """Load positive pairs for contrastive learning."""
        positive_base_path_1 = []
        positive_base_path_2 = []
        positive_chunk_num_1 = []
        positive_chunk_num_2 = []
        
        with open(self.token_pairs_json_file, 'r') as json_file:
            json_data = json.load(json_file)
            for d in json_data:
                positive_base_path_1.append(d['base_path_1'])
                positive_base_path_2.append(d['base_path_2'])
                positive_chunk_num_1.append(d['chunk_num_1'])
                positive_chunk_num_2.append(d['chunk_num_2'])
        return positive_base_path_1, positive_base_path_2, positive_chunk_num_1, positive_chunk_num_2

    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.enable_nsp or self.contrastive_mode in ['semantic', 'discourse']:
            return len(self.positive_base_path_1)
        else:
            return len(self.token_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Args:
            idx (int): Index of the sample.
        Returns:
            Depends on mode:
            - Standard: (tokens, label, modality)
            - Contrastive SimCSE: ((tokens_1, tokens_2), label, modality)
            - Contrastive Semantic/Discourse: ((sent_1, sent_2), label, modality)
            - NSP: (sent_1, sent_2, is_next, modality)
        """
        
        if self.contrastive_mode == 'simcse' and self.return_double_augmentations:
            # SimCSE-style contrastive learning: same text, different augmentations
            chunk_num = int(os.path.basename(self.token_paths[idx]).split('.npy')[0].split('_')[2])
            token_path = os.path.join(self.root_dir, '_'.join(self.token_paths[idx].split('_')[:-1]) + '.npy')

            try:
                tokens = np.load(token_path)[chunk_num]
            except FileNotFoundError:
                raise FileNotFoundError(f"Token file not found: {token_path}")

            # Create two different augmented views
            if self.augment:
                tokens_1 = self.augment_text(tokens.copy())
                tokens_2 = self.apply_stronger_augmentations(tokens.copy())
            else:
                # Even without explicit augmentation, create slight variations
                tokens_1 = self.augment_text(tokens.copy())
                tokens_2 = self.augment_text(tokens.copy())
            
            # Pad both sequences
            tokens_1 = self.pad_sequence(tokens_1)
            tokens_2 = self.pad_sequence(tokens_2)
            
            label = self.labels[idx]
            return torch.tensor(tokens_1, dtype=torch.long), torch.tensor(tokens_2, dtype=torch.long), label, self.modality
        
        elif self.contrastive_mode in ['semantic', 'discourse'] and self.return_double_augmentations:
            # Semantic/Discourse contrastive learning: related sentences as positive pairs
            base_path_1 = self.positive_base_path_1[idx]
            base_path_2 = self.positive_base_path_2[idx]
            chunk_num_1 = self.positive_chunk_num_1[idx]
            chunk_num_2 = self.positive_chunk_num_2[idx]

            # Load related sentences
            first_token_path = os.path.join(self.root_dir, base_path_1)
            first_tokens = np.load(first_token_path)
            first_token = first_tokens[chunk_num_1]

            second_token_path = os.path.join(self.root_dir, base_path_2)
            second_tokens = np.load(second_token_path)
            second_token = second_tokens[chunk_num_2]

            # Apply augmentations if enabled
            if self.augment:
                first_token = self.augment_text(first_token)
                second_token = self.augment_text(second_token)

            # Pad both sequences
            first_token = self.pad_sequence(first_token)
            second_token = self.pad_sequence(second_token)

            # Use dummy label for contrastive learning
            label = 1  # Positive pair
            return torch.tensor(first_token, dtype=torch.long), torch.tensor(second_token, dtype=torch.long), label, self.modality
        
        elif self.enable_nsp:
            # NSP mode: Next Sentence Prediction
            base_path_1 = self.positive_base_path_1[idx]
            base_path_2 = self.positive_base_path_2[idx]
            chunk_num_1 = self.positive_chunk_num_1[idx]
            chunk_num_2 = self.positive_chunk_num_2[idx]

            # Randomly choose whether to use a positive or negative pair
            use_positive = random.random() < 0.5

            if use_positive:
                is_next = 1
            else:
                random_idx = idx
                while random_idx == idx:
                    random_idx = random.randrange(len(self.positive_base_path_1))
                
                # Replace second sentence with a random one
                base_path_2 = self.positive_base_path_1[random_idx]
                chunk_num_2 = self.positive_chunk_num_1[random_idx]
                is_next = 0

            # Load sentences
            first_token_path = os.path.join(self.root_dir, base_path_1)
            first_tokens = np.load(first_token_path)
            first_token = first_tokens[chunk_num_1]

            second_token_path = os.path.join(self.root_dir, base_path_2)
            second_tokens = np.load(second_token_path)
            second_token = second_tokens[chunk_num_2]

            # Apply augmentation to first sentence
            if self.augment:
                first_token = self.augment_text(first_token)
            
            # Pad both sequences
            first_token = self.pad_sequence(first_token)
            second_token = self.pad_sequence(second_token)

            return torch.tensor(first_token), torch.tensor(second_token), is_next, self.modality
        
        else:
            # Standard mode: single text processing
            chunk_num = int(os.path.basename(self.token_paths[idx]).split('.npy')[0].split('_')[2])
            token_path = os.path.join(self.root_dir, '_'.join(self.token_paths[idx].split('_')[:-1]) + '.npy')

            try:
                tokens = np.load(token_path)[chunk_num]
            except FileNotFoundError:
                raise FileNotFoundError(f"Token file not found: {token_path}")

            # Apply augmentations if enabled
            if self.augment:
                tokens = self.augment_text(tokens)
            
            # Pad sequence
            tokens = self.pad_sequence(tokens)
            
            label = self.labels[idx]
            
            if self.return_double_augmentations:
                # Return the same sequence twice for compatibility
                return torch.tensor(tokens), torch.tensor(tokens), label, self.modality
            else:
                return torch.tensor(tokens), label, self.modality

