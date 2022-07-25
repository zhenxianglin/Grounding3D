CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch\
           --nproc_per_node=4\
           scripts/dist_train.py\
           --dist\
           --dataset scanrefer\
           --val_epoch 1\
           --batch_size 32\
           --work_dir work_dir/vil_bert3d_dist/scanrefer