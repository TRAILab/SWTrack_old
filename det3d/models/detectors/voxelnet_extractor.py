from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy
from .. import builder

@DETECTORS.register_module
class VoxelNetExtractor(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        bbox_roi_extractor,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNetExtractor, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.bbox_roi_extractor = builder.build_head(bbox_roi_extractor)

    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            prediction_dicts = self.bbox_head.predict(example, preds, self.test_cfg)
            boxes_3d = [prediction["box3d_lidar"][:, [0, 1, 2, 3, 4, 5, -1]]
                         for prediction in prediction_dicts]
            rois, roi_features = self.bbox_roi_extractor(boxes_3d, self.test_cfg)
            assert roi_features.size(dim=0) == len(prediction_dicts)
            for i in len(prediction_dicts):
                prediction_dicts[i].update(dict(roi_features=roi_features[i,:,:]))
            return prediction_dicts

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 