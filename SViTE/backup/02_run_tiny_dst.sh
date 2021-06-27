CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env main.py \
--model deit_tiny_patch16_224 \
--batch-size 64 \
--data-path ../../imagenet \
--output_dir ./tiny_dst \
--dist_url tcp://127.0.0.1:2457 \
--sparse_init fixed_ERK \
--density 0.05 \
--update_frequency 1000 \
--growth gradient \
--death magnitude \
--redistribution none