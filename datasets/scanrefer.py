import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from utils import pc_utils, scannet_utils
from utils.metrics import cal_accuracy
from pytorch_transformers.tokenization_bert import BertTokenizer
import cv2
import scipy
import random
from tqdm import tqdm

class SCANREFER(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.split = split
        if split == 'train':
            self.data_path = os.path.join(args.data_path, 'train_scanrefer.json')
        elif split == 'val':
            self.data_path = os.path.join(args.data_path, 'val_scanrefer.json')
        self.sunrefer = json.load(open(self.data_path, 'r'))
        self.sample_points_num = args.sample_points_num
        self.x_size, self.y_size = args.img_shape
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

        self.data_type = torch.float32

    def __getitem__(self, index):
        
        data = self.sunrefer[index]
        # read image
        image_path = data['image_path']
        image = scannet_utils.load_image(image_path)
        img_y_size, img_x_size, dim = image.shape
        image = cv2.resize(image, (self.x_size, self.y_size))
        image = image.transpose((2, 0, 1))

        # read points
        points = np.load(data['point_cloud_path'])[:, :3]

        # spatial
        boxes3d = np.load(f"detection_results/scannet/{data['scene_id']}/pcd_{data['object_id']}_{data['image_id']}.npy")
        objects_pc = scannet_utils.batch_extract_pc_in_box3d(points, boxes3d, self.sample_points_num)

        axis_align_matrix_path = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/{data['scene_id']}.txt"
        axis_align_matrix = scannet_utils.load_axis_align_matrix(axis_align_matrix_path)
        cam_pose_filename   = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/pose/{data['image_id']}.txt"
        cam_pose = np.loadtxt(cam_pose_filename)
        color_intrinsic_path = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/intrinsic/intrinsic_color.txt"
        color_intrinsic = np.loadtxt(color_intrinsic_path)
        bias = np.array(data['bias'])
        corners2d, corners3d = scannet_utils.batch_compute_box_3d(boxes3d, axis_align_matrix, cam_pose, color_intrinsic, bias=bias)

        spatial = []
        # for i in range(len(boxes3d)):
        #     x, _, _, l, w, h, _ = boxes3d[i]
        #     # l, w, h = l*2, w*2, h*2
        #     eachbox_2d = corners2d[i]
        #     eachbox_2d[:, 0][eachbox_2d[:, 0]<0] = 0
        #     eachbox_2d[:, 0][eachbox_2d[:, 0]>=img_x_size] = img_x_size-1
        #     eachbox_2d[:, 1][eachbox_2d[:, 1]<0] = 0
        #     eachbox_2d[:, 1][eachbox_2d[:, 1]>=img_y_size] = img_y_size-1
        #     xmin, ymin = eachbox_2d.min(axis=0)
        #     xmax, ymax = eachbox_2d.max(axis=0)
        #     square = (ymax - ymin) * (xmax - xmin) / (img_x_size * img_y_size)
        #     xmin /= img_x_size
        #     xmax /= img_x_size
        #     ymin /= img_y_size
        #     ymax /= img_y_size
        #     spatial.append([xmin, ymin, xmax, ymax, x, square])
        for i in range(len(boxes3d)):
            x, y, z, l, w, h, _ = boxes3d[i]
            spatial.append([x, y, z, l, w, h])
        spatial = np.array(spatial)

        # bboxes2d
        boxes2d = []
        for box in corners2d:
            xmin, ymin = box.min(axis=0)
            xmax, ymax = box.min(axis=0)
        #     xmin, ymin, xmax, ymax = box
            xmin = int(xmin * self.x_size / img_x_size)
            xmax = int(xmax * self.x_size / img_x_size)
            ymin = int(ymin * self.y_size / img_y_size)
            ymax = int(ymax * self.y_size / img_y_size)
            boxes2d.append([xmin, ymin, xmax, ymax])
        boxes2d = np.array(boxes2d)

        # target
        threshold = 0.25
        target = np.zeros((len(boxes3d), ), dtype=np.float32)
        target_box = np.array(data['object_box'], dtype=np.float32)
        for i, box in enumerate(boxes3d):
            iou = pc_utils.cal_iou3d(target_box, box)
            target[i] = 1 if iou >= threshold else 0

        # token
        sentence = data['sentence'].lower()
        sentence = "[CLS] " + sentence + " [SEP]"
        token = self.tokenizer.encode(sentence)

        return image, boxes2d, objects_pc, spatial, token, target
    
    def collate_fn(self, raw_batch):
        raw_batch = list(zip(*raw_batch))

        # image
        image_list = raw_batch[0]
        image = []
        for img in image_list:
            img = torch.tensor(img, dtype=self.data_type)
            image.append(img)
        image = torch.stack(image, dim=0)

        # boxes2d
        boxes2d_list = raw_batch[1]
        max_obj_num = 0
        for each_boxes2d in boxes2d_list:
            max_obj_num = max(max_obj_num, each_boxes2d.shape[0])
        
        boxes2d = torch.zeros((len(boxes2d_list), max_obj_num, 4), dtype=self.data_type)
        vis_mask = torch.zeros((len(boxes2d_list), max_obj_num), dtype=torch.long)
        for i, each_boxes2d in enumerate(boxes2d_list):
            each_boxes2d = torch.tensor(each_boxes2d)
            boxes2d[i, :each_boxes2d.shape[0], :] = each_boxes2d
            vis_mask[i, :each_boxes2d.shape[0]] = 1

        # points
        objects_pc = raw_batch[2]
        points = torch.zeros((len(boxes2d_list), max_obj_num, self.sample_points_num, 3), dtype=self.data_type)
        for i, pt in enumerate(objects_pc):
            points[i, :pt.shape[0], :, :] = torch.tensor(pt)

        # spatial
        spatial_list = raw_batch[3]
        spatial = torch.zeros((len(boxes2d_list), max_obj_num, 6), dtype=self.data_type)
        for i, box in enumerate(spatial_list):
            spatial[i, :box.shape[0], :] = torch.tensor(box)

        # token
        token_list = raw_batch[4]
        max_token_num = 0
        for each_token in token_list:
            max_token_num = max(max_token_num, len(each_token))

        token = torch.zeros((len(boxes2d_list), max_token_num), dtype=torch.long)
        for i, each_token in enumerate(token_list):
            token[i, :len(each_token)] = torch.tensor(each_token)

        mask = torch.zeros((len(boxes2d_list), max_token_num), dtype=torch.long)
        mask[token != 0] = 1

        segment_ids = torch.zeros_like(mask)

        # target
        target_list = raw_batch[5]
        target = torch.zeros((len(boxes2d_list), max_obj_num), dtype=self.data_type)
        for i, each_target in enumerate(target_list):
            target[i, :len(each_target)] = torch.tensor(each_target)       

        return image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids, target
    
    def __len__(self):
        return len(self.sunrefer)
    
    def evaluate(self, index_list):
        target_boxes = []
        pred_boxes = []
        idx = 0
        for max_index in tqdm(index_list):
            data = self.sunrefer[idx]
            boxes3d = np.load(f"detection_results/scannet/{data['scene_id']}/pcd_{data['object_id']}_{data['image_id']}.npy")
            pred_box = boxes3d[max_index]
            target = data['object_box']
            
            pred_boxes.append(pred_box)
            target_boxes.append(target)
            idx += 1
        
        target_boxes = np.array(target_boxes)
        pred_boxes = np.array(pred_boxes)

        acc25, acc50, miou = cal_accuracy(pred_boxes, target_boxes)
        return acc25, acc50, miou

    def visualize(self, args, index_list):
        import matplotlib.pyplot as plt
        base_path = args.vis_path

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        idx = 0
        for max_index in tqdm(index_list):
            data = self.sunrefer[idx]
            boxes3d = np.load(f"detection_results/scannet/{data['scene_id']}/pcd_{data['object_id']}_{data['image_id']}.npy")
            bias = np.array(data['bias'])
            boxes3d[:, :3] += bias

            pred_box = boxes3d[max_index]
            target = np.array(data['object_box'])
            target[:3] += bias

            axis_align_matrix_path = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/{data['scene_id']}.txt"
            axis_align_matrix = scannet_utils.load_axis_align_matrix(axis_align_matrix_path)
            cam_pose_filename   = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/pose/{data['image_id']}.txt"
            cam_pose = np.loadtxt(cam_pose_filename)
            color_intrinsic_path = f"/remote-home/share/ScannetForScanrefer/scans/{data['scene_id']}/intrinsic/intrinsic_color.txt"
            color_intrinsic = np.loadtxt(color_intrinsic_path)

            target_corners_2d, _ = scannet_utils.box3d_to_2d(target, axis_align_matrix, cam_pose, color_intrinsic)
            pred_corners_2d, _ = scannet_utils.box3d_to_2d(pred_box, axis_align_matrix, cam_pose, color_intrinsic)

            image_path = data['image_path']
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = scannet_utils.draw_projected_box3d(img, target_corners_2d, (255, 0, 0), thickness=2)
            img = scannet_utils.draw_projected_box3d(img, pred_corners_2d, (0, 255, 0), thickness=2)

            filepath = os.path.join(base_path, f"{idx}.jpg")
            # fig = plt.figure(figsize=())
            sentence = data['sentence']
            sentence = sentence.split()
            if len(sentence) > 8:
                sentence.insert(8, '\n')
            if len(sentence) > 16:
                sentence.insert(16, '\n')
            sentence = ' '.join(sentence)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{sentence}")
            plt.savefig(filepath, bbox_inches='tight')

            plt.clf()
            plt.close()

            idx += 1
        return