import json
from tqdm import tqdm
import numpy as np

TRAIN_PKL = json.load(open("/remote-home/linzhx/Datasets/singleRGBD/scannet_singleRGBD/SRGBD_filtered_train.json", 'r'))
VAL_PKL = json.load(open("/remote-home/linzhx/Datasets/singleRGBD/scannet_singleRGBD/SRGBD_filtered_val.json", 'r'))

def process(datatype):
    if datatype == "train":
        FILE = TRAIN_PKL
        output_name = "train_scanrefer.json"
    else:
        FILE = VAL_PKL
        output_name = "val_scanrefer.json"
    dataset = []
    for original_data in tqdm(FILE):
        scene_id = original_data['scene_id']
        object_id = int(original_data['object_id'])
        object_name = original_data['object_name']
        ann_id = original_data['ann_id']
        sentence = original_data['description']
        token = original_data['token']

        chosen_image_id_list = original_data['chosen_image_id']
        for image_id in chosen_image_id_list:
            point_cloud_path = f"/remote-home/linzhx/Projects/GD3D/detection3d/data/scannet/scannet_singleRGBD/bias_points/{scene_id}/pcd_{object_id}_{image_id}.npy"
            gt_boxes = np.load(f"/remote-home/linzhx/Projects/GD3D/detection3d/data/scannet/scannet_singleRGBD/bbox/{scene_id}/{scene_id}_bbox.npy")[:, :7]
            gt_boxes[:, 6] = 0
            object_box = gt_boxes[object_id]
            bias = np.load(f"/remote-home/linzhx/Projects/GD3D/detection3d/data/scannet/scannet_singleRGBD/bias/{scene_id}/pcd_{object_id}_{image_id}.npy")
            object_box[:3] -= bias
            object_box = object_box.tolist()
            bias = bias.tolist()
            image_path = f"/remote-home/share/ScannetForScanrefer/scans/{scene_id}/color/{image_id}.jpg"
            data = dict()
            data["scene_id"] = scene_id
            data["object_id"] = object_id
            data["image_id"] = image_id
            data["object_name"] = object_name
            data["ann_id"] = ann_id
            data["sentence"] = sentence
            data["token"] = token
            data["point_cloud_path"] = point_cloud_path
            data["object_box"] = object_box
            data["bias"] = bias
            data["image_path"] = image_path
            dataset.append(data)

    with open(output_name, 'w') as fw:  # , encoding='utf-8'
        json.dump(dataset, fw)

process('train')
process('val')
