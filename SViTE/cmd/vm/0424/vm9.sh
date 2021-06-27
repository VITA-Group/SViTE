CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --epochs 600 \
    --batch-size 128 \
    --data-path ../../imagenet \
    --output_dir ./base_dst_uns_0424_vm9 \
    --dist_url tcp://127.0.0.1:23305 \
    --sparse_init fixed_ERK \
    --density 0.3 \
    --update_frequency 6000 \
    --growth gradient \
    --death magnitude \
    --redistribution none