import torch
import torch.nn as nn
from torch.nn import functional as F

from ..registry import ROI_HEAD


@ROI_HEAD.register_module
class RoIFeatureExtractor(nn.Module):
    """Extract ROI features from BEV feature map using 3D detections."""

    def __init__(self, point_cloud_range=None):
        super(RoIFeatureExtractor, self).__init__()
        self.point_cloud_range = point_cloud_range

    def roi_grid_pool(self, rois, bev_features):
        """
        Args:
            batch_dict:
                rois (torch.Tensor): ROI coords of size (B, N, 5) in XYLWR format 
                batch_size (int): batch size
                bev_features (torch.Tensor): features of size (B, C, H, W)
        Returns:
            torch.Tensor: ROI features of size (B, N, C_out)
        """
        batch_size = rois.shape[0]
        min_x = self.point_cloud_range[0]
        min_y = self.point_cloud_range[1]
        max_x = self.point_cloud_range[3]
        max_y = self.point_cloud_range[4]
        rois_center_x = rois[:, :, 0]  # size BxN
        rois_center_y = rois[:, :, 1]  # size BxN
        rois_center_x_norm = (rois_center_x-min_x)/(max_x-min_x)*2 - 1  
        rois_center_y_norm = (rois_center_y-min_y)/(max_y-min_y)*2 - 1

        roi_features_list = []
        for b_id in range(batch_size):

            # Sample points scaled to range [-1, 1]
            
            grid_points = torch.stack((rois_center_x_norm[b_id, :], 
                                       rois_center_y_norm[b_id, :]), dim=1)  # size Nx2
            grid_points = grid_points.view(1, 1, -1, 2)  # size 1x1xNx2

            # Extract features from feature map
            feat_map = bev_features[b_id,:,:,:].unsqueeze(0)  # size 1xCxHxW
            roi_feat = F.grid_sample(feat_map, grid_points, mode='bilinear',
                                     padding_mode='zeros', align_corners=True) # size 1xCx1xN
            roi_feat = torch.squeeze(roi_feat)  # size CxN
            roi_feat = torch.transpose(roi_feat, 0, 1)  # size NxC
            roi_features_list.append(roi_feat)

        roi_features = torch.stack(roi_features_list, dim=0)
        roi_features = roi_features.view(batch_size, roi_features.shape[1], -1)

        return roi_features
    

    def boxes3d_to_rois(self, boxes_3d):
        """
        Args:
            boxes_3d (torch.Tensor): list of size B containing 
                3D boxes (N, 5) [x, y, z, l, w, h, theta] in LIDAR coordinates
        Returns:
            rois (torch.Tensor): ROI coords of size (B, N, 5) in XYLWR format 
        """
        
        rois = [boxes_batch[:, [0, 1, 3, 4, -1]] for boxes_batch in boxes_3d]
        rois = torch.stack(rois, dim=0)
        return rois


    def forward(self, boxes_3d, bev_features):
        """
        Extract ROI features from BEV feature map using 3D detections.
        Args:
            boxes_3d (list[torch.Tensor]): list of size B containing 
                3D boxes (N, 5) [x, y, z, l, w, h, theta] in LIDAR coordinates 
            bev_features (torch.Tensor): features of size (B, C, H, W)
        Returns:
            rois (torch.Tensor): ROI coords of size (B, N, 5) in XYLWR format 
            roi_features (torch.Tensor): ROI features of size (B, N, C_out)
        """
        rois = self.boxes3d_to_rois(boxes_3d)
        roi_features = self.roi_grid_pool(rois, bev_features)

        return rois, roi_features
