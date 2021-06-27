CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env main.py \
--model deit_tiny_patch16_224 \
--epochs 600 \
--batch-size 64 \
--data-path ../../imagenet \
--output_dir ./tiny_dst_uns_040501 \
--dist_url tcp://127.0.0.1:2457 \
--sparse_init fixed_ERK \
--density 0.5 \
--update_frequency 2000 \
--growth gradient \
--death magnitude \
--redistribution none