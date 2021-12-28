python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/distance_kfolddata/fold3 \
    --resume output_distance_kfoldshuffle/fold3_best/swin_large_patch4_window12_384/default/ckpt_epoch_9.pth \
    --class_type distancing \
    --version_model fold3_e9 \
    --predict_xs \