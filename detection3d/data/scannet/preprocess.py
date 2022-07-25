import json
from tqdm import tqdm
import os
import mmcv
import numpy as np
import pandas as pd


ORIGINAL_PATH = "/remote-home/share/ScannetForScanrefer/scans"
BASE_PATH = "scannet_singleRGBD"

CLASSES = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin']

TYPE2CLASS = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}

NYU40IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, \
                     19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, \
                     34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)


SUM = np.zeros((7,))
TOTAL = 0

def _get_nyu40id2class():
        lines = [line.rstrip() for line in open("meta_data/scannetv2-labels.combined.tsv")]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(TYPE2CLASS.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in NYU40IDS:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = TYPE2CLASS["others"]
                else:
                    nyu40ids2class[nyu40_id] = TYPE2CLASS[nyu40_name]
        return nyu40ids2class


def read_axis_align_matrix(meta_path):
    lines = open(meta_path).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    assert axis_align_matrix is not None
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix

def generate_frame_data(scene_id, frame_name, scene_bbox, agg_file, raw_category2nyu40ids, nyu40ids2class, axis_align_matrix):
    frame_data = dict()
    
    image_frame = frame_name[:-4].split('_')[-1]
    frame_data['image_id'] = image_frame

    # point cloud
    frame_data['point_cloud'] = dict()
    frame_data['point_cloud']['num_features'] = 6
    frame_data['point_cloud']['lidar_idx'] = scene_id

    # pts
    frame_points_path = os.path.join(BASE_PATH, 'pcd', scene_id, frame_name)
    frame_points = np.load(frame_points_path)

    XMIN, YMIN, ZMIN = frame_points[:, :3].min(axis=0)
    XMAX, YMAX, ZMAX = frame_points[:, :3].max(axis=0)
    CX = (XMIN + XMAX) / 2
    CY = (YMIN + YMAX) / 2
    points = frame_points[:, :6]
    bias = np.array([CX, CY, 0])
    points[:, :3] -= bias

    base_bias_points_path = os.path.join(BASE_PATH, 'bias_points', scene_id)
    if not os.path.exists(base_bias_points_path):
        os.makedirs(base_bias_points_path)
    bias_points_path = os.path.join(base_bias_points_path, frame_name)
    np.save(bias_points_path, points)
    frame_data['pts_path'] = bias_points_path
    # frame_data['pts_path'] = frame_points_path

    # bias
    base_bias_path = os.path.join(BASE_PATH, 'bias', scene_id)
    if not os.path.exists(base_bias_path):
        os.makedirs(base_bias_path)
    bias_path = os.path.join(base_bias_path, frame_name)
    np.save(bias_path, bias)

    # pts_instance_mask_path
    # instance_mask = frame_points[:, 7]
    base_ins_mask_path = os.path.join(BASE_PATH, 'instance', scene_id)
    # if not os.path.exists(base_ins_mask_path):
    #     os.makedirs(base_ins_mask_path)
    pts_instance_mask_path = os.path.join(base_ins_mask_path, f"{image_frame}.bin")
    # instance_mask.astype(np.int64).tofile(pts_instance_mask_path)
    frame_data['pts_instance_mask_path'] = pts_instance_mask_path

    # pts_semantic_mask_path
    # semantic_mask = frame_points[:, 6]
    # semantic_mask[semantic_mask > 40] = 0
    base_sem_mask_path = os.path.join(BASE_PATH, 'semantic', scene_id)
    # if not os.path.exists(base_sem_mask_path):
    #     os.makedirs(base_sem_mask_path)
    pts_semantic_mask_path = os.path.join(base_sem_mask_path, f"{image_frame}.bin")
    # semantic_mask.astype(np.int64).tofile(pts_semantic_mask_path)
    frame_data['pts_semantic_mask_path'] = pts_semantic_mask_path

    # anno
    frame_data['annos'] = dict()

    classlist = []
    indexlist = []
    namelist = []
    necessary_objects = []
    for i, obj_id in enumerate(np.unique(frame_points[:, 7])):
        if obj_id == 0:
            continue
        label = agg_file[int(obj_id) - 1]['label']
        label_nyu40id = raw_category2nyu40ids[label]
        if label_nyu40id not in nyu40ids2class:
            continue
        necessary_objects.append(int(obj_id)-1)
        class_id = nyu40ids2class[label_nyu40id]
        class_name = CLASSES[class_id]
        # namelist.append(class_name)
        namelist.append('objects')
        indexlist.append(i)
        # classlist.append(class_id)
        classlist.append(0)
    bboxes3d = scene_bbox[necessary_objects][:, :6]
    bboxes3d = np.insert(bboxes3d, 6, 0, axis=1)
    bboxes3d[:, :3] -= bias

    global SUM, TOTAL
    SUM = SUM + bboxes3d.sum(axis=0)
    TOTAL = TOTAL + len(bboxes3d)

    classlist = np.array(classlist, dtype=np.int64)
    indexlist = np.array(indexlist, dtype=np.int64)
    gt_num = len(namelist)
    location = bboxes3d[:, 0:3]
    dimensions = bboxes3d[:, 3:6]

    frame_data['annos']['gt_num'] = gt_num
    frame_data['annos']['name'] = namelist
    frame_data['annos']['location'] = location
    frame_data['annos']['dimensions'] = dimensions
    frame_data['annos']['gt_boxes_upright_depth'] = bboxes3d
    frame_data['annos']['axis_align_matrix'] = axis_align_matrix
    frame_data['annos']['index'] = np.array(list(range(len(classlist))), dtype=np.int64)
    frame_data['annos']['class'] = classlist
    frame_data['annos']['bias'] = bias
    return frame_data

def preprocess(datatype):
    infos = []

    nyu40ids2class = _get_nyu40id2class()

    LABEL_TSV = pd.read_csv("meta_data/scannetv2-labels.combined.tsv", sep='\t')
    raw_category = LABEL_TSV['raw_category']
    nyu40id = LABEL_TSV['nyu40id']
    raw_category = raw_category.tolist()
    nyu40id = nyu40id.tolist()
    raw_category2nyu40ids = {}
    for i in range(len(raw_category)):
        raw_category2nyu40ids[raw_category[i]] = nyu40id[i]
    
    datalist = []
    with open(f"{datatype}.txt", 'r') as f:
        for line in f.readlines():
            datalist.append(line[:-1])

    for count, scene_id in enumerate(datalist):
        print(f"Start to process [{scene_id}][{count+1}|{len(datalist)}]")

        axis_path = os.path.join(ORIGINAL_PATH, scene_id, f"{scene_id}.txt")
        axis_align_matrix = read_axis_align_matrix(axis_path)

        ins2sem = mmcv.load(os.path.join(BASE_PATH, f"ins2sem_mapping/{scene_id}/{scene_id}_ins2sem.pkl"))
        scene_bbox = np.load(os.path.join(BASE_PATH, f"bbox/{scene_id}/{scene_id}_bbox.npy"))

        agg_path = os.path.join(ORIGINAL_PATH, f"{scene_id}/{scene_id}.aggregation.json")
        agg_file = json.load(open(agg_path, 'r'))['segGroups']

        frame_list = os.listdir(os.path.join(BASE_PATH, 'pcd', scene_id))
        for frame_name in tqdm(frame_list):
            frame_data = generate_frame_data(scene_id, frame_name, scene_bbox, agg_file, raw_category2nyu40ids, nyu40ids2class, axis_align_matrix)
            infos.append(frame_data)

    if datatype == 'train':
        mmcv.dump(infos, 'scannet_det_train_2.pkl', 'pkl')
        print('scannet_det_train.pkl info test file is saved')
    elif datatype == 'val':
        mmcv.dump(infos, 'scannet_det_val_2.pkl', 'pkl')
        print('scannet_det_val.pkl info test file is saved')


preprocess('train')
preprocess('val')

MEAN = SUM / TOTAL
print(f"MEAN = {MEAN}")
