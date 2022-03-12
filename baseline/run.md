python3 baseline/scripts/train.py \
    --config baseline/configs/updown_nocaps_val.yaml \
    --config-override OPTIM.BATCH_SIZE 250 \
    --gpu-ids -1 --serialization-dir baseline/checkpoints/updown-baseline