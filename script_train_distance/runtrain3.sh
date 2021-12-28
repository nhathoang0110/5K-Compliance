python3 -m torch.distributed.launch --master_port 5000  --nproc_per_node 1 main.py \
--cfg configs/swin_large_patch4_window12_384.yaml \
--data-path /mnt/hdd10tb/Students/hoangnn/pytorch-image-models/dataset/distance_kfolddata_v1/fold3 \
--batch-size 8 \
--output output_distance_kfoldshuffle/fold3_best \
--resume swin_large_patch4_window12_384_22kto1k.pth \
--use-checkpoint \
--class_type distancing \
--opts TRAIN.EPOCHS 20 TRAIN.WARMUP_EPOCHS 3 MODEL.DROP_PATH_RATE 0.1 AUG.MIXUP 0.8 AUG.CUTMIX 1.0 TRAIN.BASE_LR 1e-3 TRAIN.WARMUP_LR 1e-6 TRAIN.MIN_LR 1e-5 \
