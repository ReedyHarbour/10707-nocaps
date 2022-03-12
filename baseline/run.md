python3 updown-baseline/scripts/train.py \
    --config updown-baseline/configs/updown_nocaps_val.yaml \
    --config-override OPTIM.BATCH_SIZE 1 \
    --gpu-ids -1 --serialization-dir updown-baseline/checkpoints/updown-baseline