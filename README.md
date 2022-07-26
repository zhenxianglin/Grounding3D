# Single-View Visual Grounding 3D
A single-view 3D visual grounding model. Detetor is VoteNet based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Grounding Model is ViLBert3D.

# Environment
Python 3.8, pytorch 1.11.0 and cuda 11.4 are used for this project.
```
conda create -n grounding3d python=3.8
conda activate grounding3d
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
```


# Install packages
Install some addition packages:
```
pip install -r requirement.txt
```
Install Pointnet++:
```
cd external_tools/pointnet2
python setup.py install
cd ../..
```

# Prepare dataset
Download datasets [here](https://github.com/UncleMEDM/Refer-it-in-RGBD).

Detection data preprocess is followed by [detection3d/data](./detection3d/data).

Run scanrefer_proprocess.py and sunrefer_process.py in [data](./data).

# Training
```
./sunrefer_dist.sh
./scanrefer_dist.sh
```
# Evaluation
```
python scripts/test.py --dataset {dataset} --eval_path {model_path} --vis_path {vis_path}
```

# Experiments Result
|        Dataset    | AP@0.25 | R@0.25 | 
|:-----------------:|:-------:|:------:|
| [SUNRefer](https://arxiv.org/pdf/2103.07894) |  0.2557 | 0.1927 |
| [ScanRefer SingleView](https://arxiv.org/pdf/2103.07894) |  0.4418 | 0.2339 |
