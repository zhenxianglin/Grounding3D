CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/dist_train.py --dist --val_epoch 1 --batch_size 64

# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/dist_train.py --dist
