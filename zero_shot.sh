########################################################################################################################
# Zero shot Script for image modality #
# Car dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29511 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/Cars \
        --image-dataset Cars \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_tune_Cars_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 196


# DTD dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29511 \
        --use_env zero_shot.py \
        --zero-shot \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/DTD \
        --image-dataset DTD \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_DTD_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 47

# EuroSAT dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29511 \
        --use_env zero_shot.py \
        --zero-shot \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/EuroSAT \
        --image-dataset EuroSAT \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_EuroSAT_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10


# GTSRB dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --zero-shot \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/GTSRB \
        --image-dataset GTSRB \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_GTSRB_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 43

# KITTI dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/KITTI \
        --image-dataset KITTI \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_KITTI_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 9

# MNIST dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/MNIST \
        --image-dataset MNIST \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_MNIST_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

# RESISC45 dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/RESISC45 \
        --image-dataset RESISC45 \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_RESISC45_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 45

# SUN397 dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SUN397 \
        --image-dataset SUN397 \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_SUN397_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 397

# SVHN dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SVHN \
        --image-dataset SVHN \
        --selected-multimodality image \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_SVHN_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10


########################################################################################################################
# Zero shot Script for audio modality #
# VGGSound dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/VGGSound \
        --audio-dataset vggsound \
        --selected-multimodality audio \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_vggsound_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 309

# EpicSound dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data1/steven/DATA/EPIC-Sounds-wav \
        --audio-dataset epicsound \
        --selected-multimodality audio \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_epicsound_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 44

# SppechCommand dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/Speech-Command-V2 \
        --audio-dataset speechcommand \
        --selected-multimodality audio \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_speechcommand_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 35

# Nsynth dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/nsynth_data \
        --audio-dataset nsynth \
        --selected-multimodality audio \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_nsynth_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 11

########################################################################################################################
# Zero Shot Script for text modality #
# AGNews dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/AG_NEWS \
        --text-dataset agnews \
        --selected-multimodality text \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_agnews_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 4

# NewsGroups20 dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/20NEWSGROUPS \
        --text-dataset newsgroups20 \
        --selected-multimodality text \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_newsgroups20_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 20

# IMDB dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/IMDB \
        --text-dataset imdb \
        --selected-multimodality text \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_imdb_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 2

# CARER dataset zero shot evaluation with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env zero_shot.py \
        --world_size 1 \
        --zero-shot \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/CARER \
        --text-dataset carer \
        --selected-multimodality text \
        --log-file zero_shot_checkpoints/train_pretrained_Omni-C_image_audio_text_zero_shot_carer_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 1 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET none DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 6

