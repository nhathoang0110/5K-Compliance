python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/distance_kfolddata/fold4 \
    --resume output_distance_kfoldshuffle/fold4_best/swin_large_patch4_window12_384/default/ckpt_epoch_10.pth \
    --class_type distancing \
    --version_model fold4_e10 \
    --predict_xs \