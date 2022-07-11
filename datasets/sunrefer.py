import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from utils import sunrgbd_utils
from pytorch_transformers.tokenization_bert import BertTokenizer
import cv2

class SUNREFER(Dataset):
    def __init__(self, args, split):
        super().__init__()
        if split == 'train':
            self.data_path = os.path.join(args.data_path, 'train_sunrefer.json')
        elif split == 'val':
            self.data_path = os.path.join(args.data_path, 'val_sunrefer.json')
        self.sunrefer = json.load(open(self.data_path, 'r'))
        self.sample_points_num = args.sample_points_num
        self.x_size, self.y_size = args.img_shape
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

        self.data_type = torch.float32

    def __getitem__(self, index):
        data = self.sunrefer[index]

        # read image
        image_path = data['image_path']
        image = sunrgbd_utils.load_image(image_path)
        img_y_size, img_x_size, dim = image.shape
        image = cv2.resize(image, (self.x_size, self.y_size))
        image = image.transpose((2, 0, 1))

        # bboxes2d
        boxes2d = []
        for box in np.array(data['boxes2d']):
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin * self.x_size / img_x_size)
            xmax = int(xmax * self.x_size / img_x_size)
            ymin = int(ymin * self.y_size / img_y_size)
            ymax = int(ymax * self.y_size / img_y_size)
            boxes2d.append([xmin, ymin, xmax, ymax])
        boxes2d = np.array(boxes2d)

        # read points
        points = np.load(data['point_cloud_path'])['pc'][:, :3]

        # spatial
        boxes3d = np.array(data['boxes'], dtype=np.float32)
        objects_pc = sunrgbd_utils.batch_extract_pc_in_box3d(points, boxes3d, self.sample_points_num)
        box_to_2d = sunrgbd_utils.batch_compute_box_3d(boxes3d, data['calib_path'])

        spatial = []
        for i in range(len(boxes3d)):
            x, _, _, l, w, h, _ = boxes3d[i]
            l, w, h = l*2, w*2, h*2
            eachbox_2d = box_to_2d[i]
            eachbox_2d[:, 0][eachbox_2d[:, 0]<0] = 0
            eachbox_2d[:, 0][eachbox_2d[:, 0]>=img_x_size] = img_x_size-1
            eachbox_2d[:, 1][eachbox_2d[:, 1]<0] = 0
            eachbox_2d[:, 1][eachbox_2d[:, 1]>=img_y_size] = img_y_size-1
            xmin, ymin = eachbox_2d.min(axis=0)
            xmax, ymax = eachbox_2d.max(axis=0)
            square = (ymax - ymin) * (xmax - xmin) / (img_x_size * img_y_size)
            xmin /= img_x_size
            xmax /= img_x_size
            ymin /= img_y_size
            ymax /= img_y_size
            spatial.append([xmin, ymin, xmax, ymax, x, square])
        spatial = np.array(spatial)
        
        # target
        object_id = int(data['object_id'])

        # token
        sentence = data['sentences'].lower()
        sentence = "[CLS] " + sentence + " [SEP]"
        token = self.tokenizer.encode(sentence)

        return image, boxes2d, objects_pc, spatial, token, object_id
    
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
        object_id_list = raw_batch[5]
        target = torch.zeros((len(boxes2d_list), max_obj_num), dtype=torch.long)
        for i, object_id in enumerate(object_id_list):
            target[i, object_id] = 1

        return image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids, target
    
    def __len__(self):
        return len(self.sunrefer)