import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils.transform import build_transform
from utils.tiny_imagenet import TinyImageNetDataset
from utils.imagenet import ImageNetDataset
from utils.cc3m import CC3MDataset
from utils.vggsound import VGGSoundDataset
from utils.agnews import AGNEWSDataset
from utils.text_dataset import TextDataset
from utils.wiki import WiKiDataset
from utils.newsgroups20 import Newsgroups20Dataset
from utils.image_dataset import ImageDataset
from utils.audio_dataset import AudioDataset
from utils.downstream_dataset import DownStreamDataset
from utils.downstream_audio_dataset import DownStreamAudioDataset

class SingleModalityDataLoader:
    def __init__(self, modality, dataset, dataset_path=None, batch_size=32, num_workers=8, shuffle=True, config=None, args=None, max_batches=None, is_train=True, is_train_img_transform=True, audio_augmentation=False, text_augmentation=False, distributed=False, num_tasks=1, rank=0, text_seq_len=512, enable_nsp=False, return_type='no_pt', return_double_augmentations=False, clip_align=False, extract_align_feature=False):
        prefix = 'train' if is_train else 'val'
        
        if dataset_path is not None:
            if modality == 'image':
                image_dataset_path = dataset_path
            elif modality == 'audio':
                audio_dataset_path = dataset_path
            elif modality == 'text':
                text_dataset_path = dataset_path
        else:
            if modality == 'image':
                image_dataset_path = config.DATA.IMAGE_DATA_PATH
                self.dataset_path = image_dataset_path
            elif modality == 'audio':
                audio_dataset_path = config.DATA.AUDIO_DATA_PATH
                self.dataset_path = audio_dataset_path
            elif modality == 'text':
                text_dataset_path = config.DATA.TEXT_DATA_PATH
                self.dataset_path = text_dataset_path
                
        # Initialize dataset and sampler
        if modality == 'image':
            image_transform = build_transform(is_train_img_transform, config)
            if dataset[modality] == 'tiny_imagenet':
                image_dataset = TinyImageNetDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'tiny_imagenet-200_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align)
            elif dataset[modality] == 'imagenet':
                image_dataset = ImageNetDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'imagenet_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align)
            elif dataset[modality] == 'cifar100':
                image_dataset = ImageDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'cifa100_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align, dataset=dataset[modality])
            elif dataset[modality] == 'iNaturalist':
                image_dataset = ImageDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'iNaturalist_User_120k_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align, dataset=dataset[modality])
            elif dataset[modality] == 'cc3m':
                image_dataset = CC3MDataset(os.path.join(image_dataset_path, 'metadata', 'cc3m_' + prefix + '_format.json'), os.path.join(image_dataset_path, 'images'), os.path.join(image_dataset_path, 'metadata'), transform=image_transform, text_seq_len=text_seq_len, clip_align=clip_align)
            elif dataset[modality] in ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN', 'KITTI', 'SUN397']:
                image_dataset = ImageDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, prefix + '.json'),  os.path.join(image_dataset_path, 'classes.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align, clip_fine_tune=True, dataset=dataset[modality])
            elif dataset[modality] in ['downstream_Cars', 'downstream_DTD', 'downstream_EuroSAT', 'downstream_GTSRB', 'downstream_MNIST', 'downstream_RESISC45', 'downstream_SUN397', 'downstream_SVHN', 'downstream_KITTI', 'downstream_SUN397']:
                image_dataset = DownStreamDataset(image_dataset_path, os.path.join(image_dataset_path, prefix + '.json'), prefix, return_type=return_type)
            else:
                raise ValueError("Unsupported image dataset")
            modality_dataset = image_dataset

        elif modality == 'audio':
            if dataset[modality] == 'vggsound':
                audio_dataset = AudioDataset(audio_dataset_path, os.path.join(audio_dataset_path, 'vggsound_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, audio_augmentation=audio_augmentation)
            elif dataset[modality] == 'audioset':
                if prefix == 'train':
                    audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'unbalanced_train_segments'), os.path.join(audio_dataset_path, 'unbalanced_' + prefix + '_segments_balanced.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, audio_augmentation=audio_augmentation)
                else:
                    audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'unbalanced_train_segments'), os.path.join(audio_dataset_path, 'unbalanced_' + prefix + '_segments_balanced.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation)
                    # audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'eval_segments'), os.path.join(audio_dataset_path,'eval_segments.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, audio_augmentation=audio_augmentation)
            elif dataset[modality] == 'audiocap':
                from utils.audiocap import AudioCapDataset
                audio_dataset = AudioCapDataset(os.path.join(audio_dataset_path, 'unbalanced_train_segments'), os.path.join(audio_dataset_path, 'audiocap_' + prefix + '.json'), config, prefix, audio_feat_dir=os.path.join(audio_dataset_path, args.saving_audio_features_folder), caption_feature_dir=os.path.join(audio_dataset_path, args.saving_caption_features_folder), clip_align=clip_align, audio_augmentation=audio_augmentation, text_seq_len=text_seq_len, extract_features=extract_align_feature)
                nb_audio_classes = 1  # Audiocap is not classification-based

            elif dataset[modality] == 'clotho':
                from utils.clotho import Clotho
                audio_dataset = Clotho(os.path.join(audio_dataset_path, prefix), os.path.join(audio_dataset_path, 'clotho_' + prefix + '.json'), config, prefix, text_seq_len=text_seq_len, audio_augmentation=audio_augmentation)
                nb_audio_classes = 1  # Audiocap is not classification-based

            elif dataset[modality] == 'epicsound':
                audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'EPIC-Sounds-100-wav-' + prefix + '-by-id'), os.path.join(audio_dataset_path,'epicsound_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping_inverse.json'), config, prefix, audio_augmentation=audio_augmentation)
            elif dataset[modality] == 'speechcommand':
                audio_dataset = AudioDataset(audio_dataset_path, os.path.join(audio_dataset_path,'speech_commands_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping_inverse.json'), config, prefix, audio_augmentation=audio_augmentation)
            elif dataset[modality] == 'nsynth':
                audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'nsynth-' + prefix, 'audio'), os.path.join(audio_dataset_path,'nsynth_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, audio_augmentation=audio_augmentation)
            elif dataset[modality] in 'downstream_vggsound':
                audio_dataset = DownStreamAudioDataset(audio_dataset_path, os.path.join(audio_dataset_path, 'vggsound_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, clip_preprocess=None)
            elif dataset[modality] == 'downstream_epicsound':
                audio_dataset = DownStreamAudioDataset(os.path.join(audio_dataset_path, 'EPIC-Sounds-100-wav-' + prefix + '-by-id'), os.path.join(audio_dataset_path,'epicsound_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, clip_preprocess=None)
            elif dataset[modality] == 'downstream_speechcommand':
                audio_dataset = DownStreamAudioDataset(audio_dataset_path, os.path.join(audio_dataset_path, 'speech_commands_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, clip_preprocess=None)
            elif dataset[modality] == 'downstream_nsynth':
                audio_dataset = DownStreamAudioDataset(os.path.join(audio_dataset_path,'nsynth-' + prefix, 'audio'), os.path.join(audio_dataset_path, 'nsynth_' + prefix + '.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, clip_preprocess=None)
            else:
                raise ValueError("Unsupported audio dataset")
            modality_dataset = audio_dataset

        elif modality == 'text':
            if dataset[modality] == 'agnews':
                text_dataset = TextDataset(os.path.join(text_dataset_path, prefix), os.path.join(text_dataset_path, 'ag_news_' + prefix + '.json'), os.path.join(text_dataset_path, 'class_mapping.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), config, prefix, text_seq_len=text_seq_len, augment=text_augmentation)
            elif dataset[modality] == 'wiki':
                text_dataset = WiKiDataset(os.path.join(text_dataset_path, 'enwiki_combined_npy'), os.path.join(text_dataset_path, 'wikitext_' + prefix + '_balanced.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), os.path.join(text_dataset_path, 'wiki_combined_metadata.json'), os.path.join(text_dataset_path, 'wiki_combined_positive_pairs_train.json'), text_seq_len=text_seq_len,  augment=text_augmentation, enable_nsp=enable_nsp, return_double_augmentations=return_double_augmentations, contrastive_mode='simcse')
            elif dataset[modality] == 'newsgroups20':
                text_dataset = TextDataset(os.path.join(text_dataset_path, prefix), os.path.join(text_dataset_path, '20newsgroups_' + prefix + '.json'), os.path.join(text_dataset_path, 'class_mapping.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), config, prefix, text_seq_len=text_seq_len, augment=text_augmentation)
            elif dataset[modality] == 'imdb':
                text_dataset = TextDataset(os.path.join(text_dataset_path, prefix), os.path.join(text_dataset_path, 'imdb_' + prefix + '.json'), os.path.join(text_dataset_path, 'class_mapping.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), config, prefix, text_seq_len=text_seq_len, augment=text_augmentation)
            elif dataset[modality] == 'carer':
                text_dataset = TextDataset(os.path.join(text_dataset_path, prefix), os.path.join(text_dataset_path, 'carer_' + prefix + '.json'), os.path.join(text_dataset_path, 'class_mapping.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), config, prefix, text_seq_len=text_seq_len, augment=text_augmentation)
            else:
                raise ValueError("Unsupported text dataset")
            modality_dataset = text_dataset

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Use DistributedSampler if distributed training is enabled
        if distributed:
            sampler = DistributedSampler(modality_dataset, num_replicas=num_tasks, rank=rank, shuffle=shuffle, drop_last=True)
            shuffle = False  # Disable shuffle in DataLoader when using DistributedSampler
        else:
            sampler = None

        # Initialize DataLoader
        self.loader = DataLoader(
            modality_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )

        if max_batches is None or max_batches >= len(self.loader):
            self.max_batches = len(self.loader)
        else:
            self.max_batches = max_batches
        print(f"{modality} dataset with {len(self.loader)} batches.")

    def __iter__(self):
        self.step = 0
        self.iterator = iter(self.loader)  # Reset iterators
        return self
    
    def __next__(self):
        if self.step >= self.max_batches:
            raise StopIteration
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset iterator for this modality if exhausted
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.step += 1
        return batch

    def __len__(self):
        return self.max_batches


# Custom Multimodal Dataloader
class MultimodalDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=8, shuffle=True, config=None, args=None, is_train=True, is_train_img_transform=True, audio_augmentation=False, text_augmentation=False, distributed=False, num_tasks=1, rank=0, text_seq_len=512, enable_nsp=False, accumulate_steps=1, selected_modalities=['image', 'audio', 'text'], return_double_augmentations=False, clip_align=False, extract_align_feature=False):
        prefix = 'train' if is_train else 'val'
        self.loaders = {}
        self.samplers = {}  # Store samplers for each modality
        self.accumulate_steps = accumulate_steps

        image_dataset_path = config.DATA.IMAGE_DATA_PATH
        audio_dataset_path = config.DATA.AUDIO_DATA_PATH
        text_dataset_path = config.DATA.TEXT_DATA_PATH
        
        if 'image' in selected_modalities:
            # Image modality
            image_transform = build_transform(is_train_img_transform, config)
            if dataset['image'] == 'tiny_imagenet':
                image_dataset = TinyImageNetDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'tiny_imagenet-200_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align)
                nb_image_classes = 200
            elif dataset['image'] == 'imagenet':
                image_dataset = ImageNetDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'imagenet_' + prefix + '.json'), os.path.join(image_dataset_path, 'class_mapping.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align)
                nb_image_classes = 1000
            elif dataset['image'] == 'cifar100':
                image_dataset = ImageDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'cifa100_' + prefix + '.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align, dataset=dataset['image'])
                nb_image_classes = 100
            elif dataset['image'] == 'iNaturalist':
                image_dataset = ImageDataset(os.path.join(image_dataset_path, prefix), os.path.join(image_dataset_path, 'iNaturalist_User_120k_' + prefix + '.json'), transform=image_transform, return_double_augmentations=return_double_augmentations, clip_align=clip_align, dataset=dataset['image'])
                nb_image_classes = 1203
            elif dataset['image'] == 'cc3m':
                image_dataset = CC3MDataset(os.path.join(image_dataset_path, 'metadata', 'cc3m_' + prefix + '_format.json'), os.path.join(image_dataset_path, 'images'), os.path.join(image_dataset_path, 'metadata'), transform=image_transform, text_seq_len=text_seq_len, clip_align=clip_align)
                nb_image_classes = 1  # CC3M is not classification-based
            else:
                raise ValueError("Unsupported image dataset")
        
            if distributed:
                image_sampler = DistributedSampler(image_dataset, num_replicas=num_tasks, rank=rank, shuffle=shuffle)
                shuffle = False  # Disable shuffle in DataLoader when using DistributedSampler
            else:
                image_sampler = None
        
            self.samplers['image'] = image_sampler
            self.loaders['image'] = DataLoader(image_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=image_sampler, pin_memory=False, drop_last=True)
        
        if 'audio' in selected_modalities:
            # Audio modality
            if dataset['audio'] == 'vggsound':
                audio_dataset = AudioDataset(audio_dataset_path, os.path.join(audio_dataset_path, 'vggsound_' + prefix + '.json'), None, config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation)
                nb_audio_classes = 309
            elif dataset['audio'] == 'audioset':
                audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'unbalanced_train_segments'), os.path.join(audio_dataset_path, 'unbalanced_' + prefix + '_segments_balanced.json'), os.path.join(audio_dataset_path, 'class_mapping.json'), config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation)
                nb_audio_classes = 527
            elif dataset['audio'] == 'epicsound':
                audio_dataset = AudioDataset(os.path.join(audio_dataset_path, 'EPIC-Sounds-100-wav-' + prefix + '-by-id'), os.path.join(audio_dataset_path,'epicsound_' + prefix + '.json'), None, config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation)
                nb_audio_classes = 44
            elif dataset['audio'] == 'speechcommand':
                audio_dataset = AudioDataset(audio_dataset_path, os.path.join(audio_dataset_path,'speech_commands_' + prefix + '.json'), None, config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation)
                nb_audio_classes = 35
            elif dataset['audio'] == 'nsynth':
                audio_dataset = AudioDataset(audio_dataset_path, os.path.join(audio_dataset_path, 'nsynth_' + prefix + '.json'), None, config, prefix, return_double_augmentations=return_double_augmentations, clip_align=clip_align, audio_augmentation=audio_augmentation, text_seq_len=text_seq_len)
                nb_audio_classes = 11
            elif dataset['audio'] == 'audiocap':
                from utils.audiocap import AudioCapDataset
                audio_dataset = AudioCapDataset(os.path.join(audio_dataset_path, 'unbalanced_train_segments'), os.path.join(audio_dataset_path, 'audiocap_' + prefix + '.json'), config, prefix, audio_feat_dir=os.path.join(audio_dataset_path, 'audiocap', args.saving_audio_features_folder), caption_feature_dir=os.path.join(audio_dataset_path, 'audiocap', args.saving_caption_features_folder), clip_align=clip_align, audio_augmentation=audio_augmentation, text_seq_len=text_seq_len, extract_features=extract_align_feature)
                nb_audio_classes = 1  # Audiocap is not classification-based
            else:
                raise ValueError("Unsupported audio dataset")
        
            if distributed:
                audio_sampler = DistributedSampler(audio_dataset, num_replicas=num_tasks, rank=rank, shuffle=shuffle)
                shuffle = False  # Disable shuffle in DataLoader when using DistributedSampler
            else:
                audio_sampler = None
        
            self.samplers['audio'] = audio_sampler
            self.loaders['audio'] = DataLoader(audio_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=audio_sampler, pin_memory=False, drop_last=True)
        
        if 'text' in selected_modalities:
            # Text modality
            if dataset['text'] == 'agnews':
                text_dataset = AGNEWSDataset(os.path.join(text_dataset_path, prefix), os.path.join(text_dataset_path, 'ag_news_' + prefix + '.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), text_seq_len=text_seq_len, augment=text_augmentation, return_double_augmentations=return_double_augmentations)
                nb_text_classes = 4
            elif dataset['text'] == 'wiki':
                text_dataset = WiKiDataset(os.path.join(text_dataset_path, 'enwiki_combined_npy'), os.path.join(text_dataset_path, 'wikitext_' + prefix + '_balanced.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), os.path.join(text_dataset_path, 'wiki_combined_metadata.json'), os.path.join(text_dataset_path, 'wiki_combined_positive_pairs_train.json'), text_seq_len=text_seq_len,  augment=text_augmentation, enable_nsp=enable_nsp, return_double_augmentations=return_double_augmentations, contrastive_mode='simcse')
                nb_text_classes = 1  # Wiki dataset is not classification-based
            elif dataset['text'] == 'newsgroups20':
                text_dataset = Newsgroups20Dataset(text_dataset_path, os.path.join(text_dataset_path, '20newsgroups_' + prefix + '.json'), os.path.join(text_dataset_path, 'bert_vocab.json'), text_seq_len=text_seq_len, augment=text_augmentation, return_double_augmentations=return_double_augmentations)
                nb_text_classes = 20
            else:
                raise ValueError("Unsupported text dataset")
            
            if distributed:
                text_sampler = DistributedSampler(text_dataset, num_replicas=num_tasks, rank=rank, shuffle=shuffle)
                shuffle = False  # Disable shuffle in DataLoader when using DistributedSampler
            else:
                text_sampler = None
            
            self.samplers['text'] = text_sampler
            self.loaders['text'] = DataLoader(text_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=text_sampler, pin_memory=False, drop_last=True)
            
        # Initialize iterators and modalities
        all_modalities_order = ["image", "audio", "text"]  # Fixed order: image -> audio -> text
        self.modalities = [mod for mod in all_modalities_order if mod in selected_modalities and mod in self.loaders]

        self.iterators = {mod: iter(loader) for mod, loader in self.loaders.items()}
        self.batch_size = batch_size

        for modality, loader in self.loaders.items():
            print(f"{modality} dataset with {len(loader)} batches.")
        self.max_batches_per_modality = min(len(loader) for loader in self.loaders.values())  # Ensure equal steps
        print(f"Max batches per modality: {self.max_batches_per_modality}")
        

    def __iter__(self):
        self.step = 0
        self.iterators = {mod: iter(loader) for mod, loader in self.loaders.items()}  # Reset iterators
        return self
    
    def __next__(self):
        if self.step >= self.max_batches_per_modality * len(self.modalities):
            raise StopIteration
        
        # Determine modality based on step
        modality_idx = (self.step // self.accumulate_steps) % len(self.modalities)
        modality = self.modalities[modality_idx]
        
        try:
            batch = next(self.iterators[modality])
        except StopIteration:
            # Reset iterator for this modality if exhausted
            self.iterators[modality] = iter(self.loaders[modality])
            batch = next(self.iterators[modality])
        
        self.step += 1
        return batch
    
    def __len__(self):
        # Total batches = min batches per modality * 3 modalities
        return self.max_batches_per_modality * len(self.modalities)