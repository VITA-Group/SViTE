CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env main.py \
--model deit_tiny_patch16_224 \
--epochs 600 \
--batch-size 128 \
--data-path ../../../imagenet \
--output_dir ./tiny_structure \
--dist_url tcp://127.0.0.1:2454 \
--sparse_init fixed_ERK \
--update_frequency 1000 \
--growth gradient \
--death magnitude \
--redistribution none \
--atten_head 3 \
--pruning_type structure