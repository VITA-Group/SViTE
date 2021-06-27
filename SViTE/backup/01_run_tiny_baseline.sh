CUDA_VISIBLE_DEVICES=$1 \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--model deit_tiny_patch16_224 \
--batch-size 256 \
--data-path ../../../imagenet \
--output_dir ./tiny_baseline \
--dist_url tcp://127.0.0.1:3333