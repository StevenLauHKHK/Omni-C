########################################################################################################################
# Linear Probe Script for image modality #

# Car dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/Cars \
        --image-dataset Cars \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_Cars_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_Cars_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
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

# DTD dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/DTD \
        --image-dataset DTD \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_DTD_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_DTD_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 47

# EuroSAT dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/EuroSAT \
        --image-dataset EuroSAT \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_EuroSAT_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_EuroSAT_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

# GTSRB dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/GTSRB \
        --image-dataset GTSRB \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_GTSRB_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_GTSRB_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 43

# KITTI dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/KITTI \
        --image-dataset KITTI \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_KITTI_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_KITTI_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 9

# MNIST dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/MNIST \
        --image-dataset MNIST \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_MNIST_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_MNIST_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

# RESISC45 dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/RESISC45 \
        --image-dataset RESISC45 \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_RESISC45_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_RESISC45_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 45

# SUN397 dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SUN397 \
        --image-dataset SUN397 \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_SUN397_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_SUN397_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 397

# SVHN dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=28531 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality image \
        --image-data-path /data1/DATA/SVHN \
        --image-dataset SVHN \
        --selected-multimodality image \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_SVHN_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_SVHN_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 10

########################################################################################################################
# Linear Probe Script for audio modality #
# VGGSound dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/VGGSound \
        --audio-dataset vggsound \
        --selected-multimodality audio \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_vggsound_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_vggsound_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 309

# EpicSound dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data1/steven/DATA/EPIC-Sounds-wav \
        --audio-dataset epicsound \
        --selected-multimodality audio \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_epicsound_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_epicsound_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 44

# SppechCommand dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/Speech-Command-V2 \
        --audio-dataset speechcommand \
        --selected-multimodality audio \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_speechcommand_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_speechcommand_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 35

# Nsynth dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29357 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality audio \
        --audio-data-path /data_ssd/DATA/nsynth_data \
        --audio-dataset nsynth \
        --selected-multimodality audio \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_nsynth_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_nsynth_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 11

########################################################################################################################
# Linear Probe Script for text modality #
# AGNews dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/AG_NEWS \
        --text-dataset agnews \
        --selected-multimodality text \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_agnews_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_agnews_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 4

# NewsGroups20 dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/20NEWSGROUPS \
        --text-dataset newsgroups20 \
        --selected-multimodality text \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_newsgroups20_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_newsgroups20_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 20

# IMDB dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/IMDB \
        --text-dataset imdb \
        --selected-multimodality text \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_imdb_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_imdb_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 2

# CARER dataset linear probe with ViT-B-32 image encoder trained with SSL on image, audio, and text modalities. 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29512 \
        --use_env linear_probe.py \
        --world_size 1 \
        --model ViT-B-32 \
        --model-size base \
        --single-modality text \
        --text-data-path /data_ssd/DATA/CARER \
        --text-dataset carer \
        --selected-multimodality text \
        --checkpoint-dir linear_probe_checkpoints/checkpoint_pretrained_Omni-C_image_audio_text_linear_probe_carer_bs128_ddp \
        --log-file linear_probe_checkpoints/train_pretrained_Omni-C_image_audio_text_linear_probe_carer_bs128_ddp.txt \
        --resume-checkpoint /data2/steven/MAE_25072025/pretrained_checkpoints/checkpoint_mix_modality_vit_base_ps32_image_audio_text_ssl_bs256_ddp_adjust_lr_1e-4_retrain_audio_length_256_and_adjust_scheduler_v3_split_projector/checkpoint-99.pth \
        --batch-size 128 \
        --accum-iter 1 \
        --save_every_n 5 \
        --eval_every_n 1 \
        --early_stop_patience 105 \
        --opts TRAIN.EPOCHS 40 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.LR 1e-4 TRAIN.OPTIMIZER.MIN_LR 1e-5 TRAIN.OPTIMIZER.WEIGHT_DECAY 0.1 TRAIN.LR_SCHEDULER.WARMUP_EPOCHS 10 \
        TEXT.TASK simcse TEXT.ENABLE_NSP False TEXT.MAX_SEQ_LENGTH 256 \
        AUDIO.MELBINS 128 AUDIO.TARGET_LENGTH 256 AUDIO.FREQM 25 AUDIO.TIMEM 50 \
        AUG.PRESET weak DATA.IMG_SIZE 224 \
        DATA.NUM_CLASSES 6
