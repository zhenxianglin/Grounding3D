CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 scripts/dist_train.py --dist --val_epoch 10
# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/dist_train.py --dist
