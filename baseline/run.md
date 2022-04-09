python baseline/scripts/train.py \
    --config baseline/configs/updown_nocaps_val.yaml \
    --config-override OPTIM.BATCH_SIZE 32 \
    --gpu-ids -1 --serialization-dir baseline/checkpoints/updown-baseline

python baseline/scripts/inference.py \
    --config baseline/configs/updown_nocaps_test.yaml \
    --checkpoint-path baseline/checkpoints/updown-baseline/checkpoint_10000.pth \
    --output-path predictions.json \
    --gpu-ids -1