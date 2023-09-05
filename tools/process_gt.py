import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm
try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("No Tensorflow")
import torch

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
CUR_TYPE_LIST = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}
ID_TO_CAT_NAME = {
    1: 'VEHICLE',
    2: 'PEDESTRIAN',
    3: 'SIGN',
    4: 'CYCLIST'
}
CUSTOM_NAME_TO_ID = {
    'VEHICLE': 0,
    'PEDESTRIAN': 1,
    'CYCLIST': 2
}

def get_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj 

def sort_frame(frames):
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def get_available_frames(root, split):
    dir_path = os.path.join(root, split, 'lidar')
    available_frames = list(os.listdir(dir_path))

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(available_frames))
    return sorted_frames

def get_infos(root_path, frames, split='val', nsweeps=2):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        lidar_path = os.path.join(root_path, split, 'lidar', frame_name)
        ref_path = os.path.join(root_path, split, 'annos', frame_name)

        ref_obj = get_obj(ref_path)
        ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        _, ref_from_global = veh_pos_to_transform(ref_pose)

        info = {
            "path": lidar_path,
            "anno_path": ref_path, 
            "token": frame_name,
            "timestamp": ref_time,
            "sweeps": []
        }

        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4]) # remove .pkl

        prev_id = frame_id
        sweeps = [] 
        while len(sweeps) < nsweeps - 1:
            if prev_id <= 0:
                if len(sweeps) == 0:
                    sweep = {
                        "path": lidar_path,
                        "token": frame_name,
                        "transform_matrix": None,
                        "time_lag": 0
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_id = prev_id - 1
                # global identifier  

                curr_name = 'seq_{}_frame_{}.pkl'.format(sequence_id, prev_id)
                curr_lidar_path = os.path.join(root_path, split, 'lidar', curr_name)
                curr_label_path = os.path.join(root_path, split, 'annos', curr_name)
                
                curr_obj = get_obj(curr_label_path)
                curr_pose = np.reshape(curr_obj['veh_to_global'], [4, 4])
                global_from_car, _ = veh_pos_to_transform(curr_pose) 
                
                tm = reduce(
                    np.dot,
                    [ref_from_global, global_from_car],
                )

                curr_time = int(curr_obj['frame_name'].split("_")[-1])
                time_lag = ref_time - 1e-6 * curr_time

                sweep = {
                    "path": curr_lidar_path,
                    "transform_matrix": tm,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        if split != 'test':
            # read boxes 
            TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
            annos = ref_obj['objects']
            num_points_in_gt = np.array([ann['num_points'] for ann in annos])
            gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
            
            if len(gt_boxes) != 0:
                # transform from Waymo to KITTI coordinate 
                # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
                # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
                gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
                gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

            gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

            # filter boxes without lidar points 
            info['gt_boxes'] = gt_boxes[mask_not_zero, :].astype(np.float32)
            info['gt_names'] = gt_names[mask_not_zero].astype(str)

        infos.append(info)
    return infos

def veh_pos_to_transform(veh_pos):
    "convert vehicle pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

def process_gt_detection(infos, tracking=True):
    """Creates a gt prediction object file for local evaluation."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    
    objects = metrics_pb2.Objects()
    gt_dict = {}
    for idx in tqdm(range(len(infos))): 
        info = infos[idx]
        token = info['token']
        obj = get_obj(info['anno_path'])
        annos = obj['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        box3d = np.array([ann['box'] for ann in annos])

        if len(box3d) == 0:
            print(f'{token} has no detections')
            gt_dict[token] = {
                'tracking_ids': [],
                'box3d_lidar': [],
                'label_preds': torch.Tensor([]),
                'scores': []
            }
            continue

        names = np.array([TYPE_LIST[ann['label']] for ann in annos])
        processed_annos = []
        for x in annos:
            if ID_TO_CAT_NAME[x['label']] in CUR_TYPE_LIST:
                x['label'] = CUSTOM_NAME_TO_ID[ID_TO_CAT_NAME[x['label']]]
                processed_annos.append(x)
        annos = processed_annos
        tracking_ids = []
        box3d_lidar = torch.Tensor(len(annos), 9)
        label_preds = []
        scores = torch.full((len(annos),), 1.0, dtype=torch.float)
        for i in range(len(annos)):
            tracking_ids.append(annos[i]['name'])
            label_preds.append(annos[i]['label'])
            box3d_lidar[i] = torch.from_numpy(annos[i]['box'])
        label_preds = torch.Tensor(label_preds)
        cur_det = {
            'tracking_ids': tracking_ids,
            'box3d_lidar': box3d_lidar,
            'label_preds': label_preds,
            'scores': scores
        }
        gt_dict[token] = cur_det
    
    gt_pkl_path = 'data/Waymo/gt_preds_test.pkl'
    with open(gt_pkl_path, 'wb') as f:
        pickle.dump(gt_dict, f)

    print("Dictionary saved to", gt_pkl_path)

if __name__ == '__main__':
    root_path = 'data/Waymo'
    pkl_path = 'data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl'
    split = 'val'
    frames = get_available_frames(root_path, split)
    infos = get_infos(root_path=root_path, frames=frames)
    process_gt_detection(infos)
    
    