import json
import pickle
import sys
import numpy as np

from tqdm import tqdm

sys.path.append("../utils")
import sunrgbd_utils as utils


# sunrefer = mmcv.load("/remote-home/linzhx/Projects/Refer-it-in-RGBD/docs/SUNREFER_v2.json")

root_box2d_path = "/remote-home/linzhx/Datasets/SUNREFER/sunrgbd_pc_bbox_votes_30k_v2/{}_bbox2d.npy"
root_box_path = "/remote-home/linzhx/Datasets/SUNREFER/sunrgbd_pc_bbox_votes_30k_v2/{}_bbox.npy"
root_pc_path = "/remote-home/linzhx/Datasets/SUNREFER/sunrgbd_pc_bbox_votes_30k_v2/{}_pc.npz"
root_cali_path = "/remote-home/linzhx/Projects/Refer-it-in-RGBD/sunrgbd/sunrgbd_trainval/calib/{}.txt"
root_img_path = "/remote-home/linzhx/Projects/Refer-it-in-RGBD/sunrgbd/sunrgbd_trainval/image/{}.jpg"
root_label_path = "/remote-home/linzhx/Projects/Refer-it-in-RGBD/sunrgbd/sunrgbd_trainval/label/{}.txt"

# train_set = set()
# val_set = set()

train_pkl = pickle.load(open("/remote-home/linzhx/Datasets/SUNREFER/SUNREFER_train.pkl", 'rb'))
val_pkl = pickle.load(open("/remote-home/linzhx/Datasets/SUNREFER/SUNREFER_val.pkl", 'rb'))
# print(len(train_pkl), len(val_pkl), len(train_pkl)+len(val_pkl), len(sunrefer))
# for data in train_pkl:
#     if data["image_id"] == "050010":
#         print("Train")
#     train_set.add(data['image_id'])
# for data in val_pkl:
#     if data["image_id"] == "050010":
#         print("Val")
#     val_set.add(data['image_id'])
# exit()

def process(sunrefer, path):
    new_sunrefer = []
    for data in tqdm(sunrefer):
        image_id = data['image_id']
        object_id = int(data['object_id'])
        label_path = root_label_path.format(image_id)
        objects_list = utils.read_sunrgbd_label(label_path)

        box_path = root_box_path.format(image_id)
        box2d_path = root_box2d_path.format(image_id)
        boxes = np.load(box_path)
        boxes2d = np.load(box2d_path)

        object_box = boxes[object_id]
        point_cloud_path = root_pc_path.format(image_id)
        calib_path = root_cali_path.format(image_id)
        image_path = root_img_path.format(image_id)

        labels = [obj.classname for obj in objects_list]

        data['boxes'] = boxes.tolist()
        data['object_box'] = object_box.tolist()
        data['point_cloud_path'] = point_cloud_path
        data['calib_path'] = calib_path
        data['image_path'] = image_path
        data['labels'] = labels
        data['boxes2d'] = boxes2d.tolist()
        try:
            data['sentences'] = data['sentences']
        except:
            data['sentences'] = data['sentence']
        # print(data)
        new_sunrefer.append(data)

        # print(val_sunrefer)
        # exit()
    with open(path, 'w') as fw:  # , encoding='utf-8'
        json.dump(new_sunrefer, fw)

process(val_pkl, 'val_sunrefer.json')
process(train_pkl, 'train_sunrefer.json')



# labelset = set()

# for data in tqdm(val_pkl+train_pkl):
#     try:
#         label = data['object_name']
#         # print(label)
#         # break
#         labelset.add(label)
#     except:
#         # print(data['image_id'])
#         pass

# print(labelset)
# print(len(labelset))
