import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from .backbone.pointnet import PointNetPP
from .backbone.resnet import ResNet


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, linear_layer_num, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.layer_num = linear_layer_num

        self.linear= nn.ModuleList()
        for i in range(self.layer_num):
            if i == 0:
                self.linear.append(nn.Linear(in_size, out_size))
            else:
                self.linear.append(nn.Linear(out_size, out_size))
            if i != (self.layer_num - 1):
                self.linear.append(nn.ReLU())
                self.linear.append(nn.Dropout(self.dropout))
            
    def forward(self, x):
        for block in self.linear:
            x = block(x)
        return x

class PointCloudExtactor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pointnet = PointNetPP(sa_n_points=[32, 16, 16],
                                   sa_n_samples=[32, 32, 32],
                                   sa_radii=[0.2, 0.4, 0.4],
                                   sa_mlps=[[0, 64, 64, 128],
                                            [128, 128, 128, 256],
                                            [256, 256, 512, args.pc_out_dim]])
        if args.pc_out_dim != args.vis_out_dim:
            self.linear = nn.Sequential(
                LinearBlock(args.pc_out_dim, args.vis_out_dim, args.linear_layer_num, args.dropout),
                nn.LayerNorm(args.vis_out_dim)
            )
        else:
            self.linear = nn.Identity()

    def forward(self, points):
        point_cloud_feature = self.pointnet.get_siamese_features(points, aggregator=torch.stack)
        point_cloud_feature = self.linear(point_cloud_feature)
        return point_cloud_feature

class ImageExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = ResNet(args)
        self.output_dim = self.feature_extractor.output_dim
        self.roi_align = RoIAlign(args.roi_size, spatial_scale=args.spatial_scale, sampling_ratio=-1)
        
        if args.objects_aggregator == 'pool':
            self.aggegator = nn.AdaptiveAvgPool1d(output_size=1)
        elif args.objects_aggregator == 'linear':
            self.aggegator = nn.Linear(args.roi_size[0]*args.roi_size[1], 1)
        else:
            raise ValueError("Wrong objects aggregator")
        
        if self.output_dim != args.vis_out_dim:
            self.linear = nn.Sequential(
                LinearBlock(self.output_dim, args.vis_out_dim, args.linear_layer_num, args.dropout),
                nn.LayerNorm(args.vis_out_dim)
            )
        else:
            self.linear = nn.Identity()

    def extract_box_feature(self, image_feature, boxes):
        object_boxes = []
        B, N, _ = boxes.shape
        for i in range(B):
            object_boxes.append(boxes[i])
        objects_feature = self.roi_align(image_feature, object_boxes)
        objects_feature = objects_feature.view(objects_feature.shape[0], objects_feature.shape[1], -1)
        objects_feature = self.aggegator(objects_feature).view(B, N, self.output_dim)
        return objects_feature
    
    def forward(self, image, boxes):
        image_feature = self.feature_extractor(image)
        objects_feature = self.extract_box_feature(image_feature, boxes)
        objects_feature = self.linear(objects_feature)
        return objects_feature

class PointcloudImageFusion(nn.Module):
    def __init__(self, args, image_out_dim):
        super().__init__()
        self.linear = nn.Sequential(
                LinearBlock(args.vis_out_dim * 2, args.vis_out_dim, args.linear_layer_num, args.dropout),
                nn.LayerNorm(args.vis_out_dim)
            )
    
    def forward(self, image_feature, point_cloud_feature):
        visual_feature = torch.cat([image_feature, point_cloud_feature], dim=2)
        visual_feature = self.linear(visual_feature)
        return visual_feature

class ViLBert3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_cloud_extractor = PointCloudExtactor(args)
        self.image_extractor = ImageExtractor(args)
        self.fusion = PointcloudImageFusion(args, self.image_extractor.output_dim)
    
    def forward(self, image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids):
        # extract point cloud feature
        point_cloud_feature = self.point_cloud_extractor(points)
        print("point_cloud_feature: ", point_cloud_feature.shape)

        # extract image feature
        image_feature = self.image_extractor(image, boxes2d)
        print("image_feature: ", image_feature.shape)

        # fusion
        visual_feature = self.fusion(image_feature, point_cloud_feature)
        print("visual_feature: ", visual_feature.shape)


        return 
