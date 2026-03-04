# Omni-C: Compressing Heterogeneous Modalities into a Single Dense Encoder

This repository implements the model proposed in the paper:

Kin Wai Lau, Yasar Abbas Ur Rehman, Lai-Man Po, Pedro Porto Buarque de Gusmão **Omni-C: Compressing Heterogeneous Modalities into a Single Dense Encoder**

[[arXiv paper]](Updated Later)


## Citing

When using this code, kindly reference:

```

```

## TODO

- [x] Upload the pretraining code
- [x] Upload the linear probe and SBoRA fine tuning code  
- [x] Upload the zero-shot code and eval code
- [x] Upload the pretraining weights
- [ ] Upload alignment code and eval code


## Preparation
```
conda env create -f environment.yml
```

## Pretrained models
Omni-C (Image, Audio and Text) [link](https://1drv.ms/f/c/92bd6f3465cb151e/IgDdJoS1x2iLSbJWCD_V1jwZAWGa15jQAFGEoKLPvyWWjws?e=h5E3iW)

## Pre-training dataset
* ImageNet-1K:
  -URL of the dataset [link](https://www.image-net.org/download.php)

* AudioSet:
  -URL of the dataset [link](https://research.google.com/audioset/)

* English Wikipedia:
  -URL of the dataset [link](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701)
  -The binary tokenize file can be generated via (data_preprocess/wiki_cleanup.py)
  -The whole preprocessed dataset can be download [link](https://1drv.ms/f/c/92bd6f3465cb151e/IgBZxV1yDtIDTZrb041c8N7-AZkBfYtzjZ-sGwCSFCrsSQM?e=goP6FC)

## Pretraining dataset annotation json file
The dataset annotation json file should have the following format:
```
[
  {
    "name": /path/to/image1
    "label": label1
  },
  {
    "name": /path/to/image2
    "label": label2
  },
]
```
The json file can be download via [link](https://1drv.ms/f/c/92bd6f3465cb151e/IgAFvJYlI11_Sb4ujXO1yqChAeCi-Fw6_oUjzpWZfL3qPRg?e=ReUas8)


## Pre-training script
Run the following command to start pre-training (see pretraining.sh):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=29503 \
        --use_env pretraining.py \
        --world_size 8 \
        --simclr \
        --image-data-path /path/to/image/imagenet1k \
        --image-dataset imagenet \
        --audio-data-path /path/to/audio/audioset \
        --audio-dataset audioset \
        --text-data-path /path/to/text/wikipedia \
        --text-dataset wiki \
        --selected-multimodality image audio text \
        --model ViT-B-32 \
        --model-size base \
        --checkpoint-dir /path/to/checkpoints
        --debug-dir /path/to/debug_dir
        --log-file /path/to/logs/training.log
        --batch-size 32 \
        --accum-iter 1 \
        --save_every_n 5 \
        --opts TRAIN.EPOCHS 100 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-6 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 5 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUG.PRESET simclr DATA.IMG_SIZE 224 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50
```
### Key Parameters
#### Data Configuration
- `--image-data-path`: Path to ImageNet-1K dataset
- `--audio-data-path`: Path to AudioSet dataset  
- `--text-data-path`: Path to English Wikipedia dataset
- `--selected-multimodality`: Modalities to use (image, audio, text)

#### Model Configuration
- `--model`: Model architecture (ViT-B-32)
- `--model-size`: Model size (base)
- `--batch-size`: Batch size per GPU (32)

#### Training Configuration
- `--world_size`: Number of GPUs (8)
- `TRAIN.EPOCHS`: Number of training epochs (100)
- `TRAIN.OPTIMIZER.LR`: Learning rate (1e-4)
- `TRAIN.OPTIMIZER.WEIGHT_DECAY`: Weight decay (0.1)
- `TRAIN.LR_SCHEDULER.WARMUP_EPOCHS`: Warmup epochs (5)

#### Output Configuration
- `--checkpoint-dir`: Directory to save model checkpoints
- `--debug-dir`: Directory for debug outputs
- `--log-file`: Path to training log file
- `--save_every_n`: Save checkpoint every N epochs (5)

### Customization
To modify the training configuration:
1. Update dataset paths to match your local setup
2. Adjust batch size based on GPU memory
3. Modify learning rate and other hyperparameters as needed
4. Change output directories for checkpoints and logs

##  Linear probe script
Run the following command to start linear probe (see linear_probe.sh):
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /path/to/downstream/dataset \
        --image-dataset Cars \
        --selected-multimodality image \
        --checkpoint-dir /path/to/linear_probe_checkpoints \
        --log-file /path/to/logs/linear_probe.log \
        --resume-checkpoint /path/to/pretrained/checkpoint.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196
```

### Key Parameters
#### Model Configuration
- `--model`: Model architecture (ViT-B-32)
- `--model-size`: Model size (base)
- `--resume-checkpoint`: Path to pre-trained Omni-C checkpoint
- `--single-modality`: Modality to evaluate (image/audio/text)
- `--selected-multimodality`: Selected modalities (should match training)

#### Data Configuration
- `--image-data-path`: Path to downstream dataset
- `--image-dataset`: Dataset name (Cars, DTD, etc.)
- `DATA.NUM_CLASSES`: Number of classes in downstream dataset
- `DATA.IMG_SIZE`: Input image size (224)

#### Training Configuration
- `--batch-size`: Batch size (128)
- `--world_size`: Number of GPUs (1 for single GPU)
- `TRAIN.EPOCHS`: Number of training epochs (40)
- `TRAIN.OPTIMIZER.LR`: Learning rate (1e-4)
- `TRAIN.OPTIMIZER.WEIGHT_DECAY`: Weight decay (0.1)
- `TRAIN.LR_SCHEDULER.WARMUP_EPOCHS`: Warmup epochs (10)

#### Output Configuration
- `--checkpoint-dir`: Directory to save linear probe checkpoints
- `--log-file`: Path to training log file
- `--save_every_n`: Save checkpoint every N epochs (5)
- `--eval_every_n`: Evaluate every N epochs (1)
- `--early_stop_patience`: Early stopping patience (105)

##  SBoRA Fine Tuning script
Run the following command to start SBoRA fine tuning (see fine_tune_sbora.sh):
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29412 \
        --use_env fine_tune_sbora.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /path/to/downstream/dataset \
        --image-dataset Cars \
        --selected-multimodality image \
        --checkpoint-dir /path/to/sbora_checkpoints \
        --log-file /path/to/logs/sbora_fine_tune.log \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --sbora-rank 128 \
        --sbora-alpha 128 \
        --sbora-dropout 0.0 \
        --sbora-mode FA \
        --freeze-patch-embedding \
        --sbora-base-checkpoint /path/to/pretrained/checkpoint.pth \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196
```
### Key Parameters

#### Model Configuration
- `--model`: Model architecture (ViT-B-32)
- `--model-size`: Model size (base)
- `--sbora-base-checkpoint`: Path to pre-trained Omni-C checkpoint
- `--single-modality`: Modality to fine-tune (image/audio/text)
- `--selected-multimodality`: Selected modalities (should match training)

#### SBoRA Specific Parameters
- `--sbora-rank`: Rank of SBoRA adaptation (128)
- `--sbora-alpha`: Alpha parameter for SBoRA (128)
- `--sbora-dropout`: Dropout rate for SBoRA layers (0.0)
- `--sbora-mode`: SBoRA mode (FA - Freeze Matrix A and update Matrix B, FB - Freeze Matrix B and update Matrix A)
- `--freeze-patch-embedding`: Freeze patch embedding layers

#### Data Configuration
- `--image-data-path`: Path to downstream dataset
- `--image-dataset`: Dataset name (Cars, DTD, etc.)
- `DATA.NUM_CLASSES`: Number of classes in downstream dataset
- `DATA.IMG_SIZE`: Input image size (224)

#### Training Configuration
- `--batch-size`: Batch size (128)
- `--world_size`: Number of GPUs (1 for single GPU)
- `TRAIN.EPOCHS`: Number of training epochs (40)
- `TRAIN.OPTIMIZER.LR`: Learning rate (1e-4)
- `TRAIN.OPTIMIZER.WEIGHT_DECAY`: Weight decay (0.1)
- `TRAIN.LR_SCHEDULER.WARMUP_EPOCHS`: Warmup epochs (10)

#### Output Configuration
- `--checkpoint-dir`: Directory to save SBoRA fine-tuned checkpoints
- `--log-file`: Path to training log file
- `--save_every_n`: Save checkpoint every N epochs (5)
- `--eval_every_n`: Evaluate every N epochs (1)
- `--early_stop_patience`: Early stopping patience (105)


##  Zero-shot eval script
Run the following command to start zero-shot eval on pretrained model (see zero_shot.sh):
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29511 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /path/to/evaluation/dataset \
        --image-dataset Cars \
        --selected-multimodality image \
        --log-file /path/to/logs/zero_shot_eval.log \
        --resume-checkpoint /path/to/pretrained/checkpoint.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196
```
### Key Parameters

#### Model Configuration
- `--model`: Model architecture (ViT-B-32)
- `--model-size`: Model size (base)
- `--resume-checkpoint`: Path to pre-trained Omni-C checkpoint
- `--zero-shot`: Enable zero-shot evaluation mode
- `--single-modality`: Modality to evaluate (image/audio/text)
- `--selected-multimodality`: Selected modalities (should match training)

#### Data Configuration
- `--image-data-path`: Path to evaluation dataset
- `--image-dataset`: Dataset name (Cars, CIFAR-10, ImageNet, etc.)
- `DATA.NUM_CLASSES`: Number of classes in evaluation dataset
- `DATA.IMG_SIZE`: Input image size (224)
- `AUG.PRESET`: Data augmentation preset (none for zero-shot)

#### Evaluation Configuration
- `--batch-size`: Batch size for evaluation (128)
- `--world_size`: Number of GPUs (1 for single GPU)
- `--log-file`: Path to evaluation log file


