python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --data_predict data/ \
    --predict \
    --batch-size 8 \
    --cfg configs/swin_large_patch4_window12_384.yaml \
    --data-path dataset/distance_kfolddata/fold2 \
    --resume output_distance_kfoldshuffle/fold2_best/swin_large_patch4_window12_384/default/ckpt_epoch_11.pth \
    --class_type distancing \
    --version_model fold2_e11 \
    --predict_xs \