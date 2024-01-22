import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import os
from utils import *
import glob
from joblib import Parallel, delayed
import multiprocessing
from progressbar import ProgressBar
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import math
from numpy.linalg import inv
import torch

def read_tfrecord(path):
    dataset = tf.data.TFRecordDataset(path)
    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the feature description
        feature_description = {
            'feature_1': tf.io.FixedLenFeature([], tf.string)
        }
        return tf.io.parse_single_example(example_proto, feature_description)
    parsed_dataset = dataset.map(_parse_function)
    for parsed_example in parsed_dataset:
        print(parsed_example)

def rotation_matrix_to_euler_angles(rotation_matrix):
    # Calculate the rotation angles
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    if sy < 1e-6:
        # Singular case: sy is close to zero
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0.0
    else:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    # Convert angles to degrees
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)
    
    return [x, y, z]


def get_total_transform(ego_pose_dict, idx):
    total_rot = np.eye(3)
    total_translation = np.array([0., 0., 0.]) 
    for i, p in enumerate(ego_pose_dict.values()):
        if i <= idx:
            total_rot = p[:3, :3] #np.dot(total_rot, p[:3,:3])
            total_translation = p[:3, 3]
    total_rot_angles = rotation_matrix_to_euler_angles(total_rot)
    return total_rot, total_translation

def lidar_top_view(predictions,
                   frame,
                   range_images,
                   camera_projections,
                   range_image_top_pose,
                   token,
                   out_file,
                   track_file,
                   frame_id,
                   last_frame,
                   seq_start_idx,
                   seq_end_idx,
                   gt,
                   infos,
                   idx,
                   ego_pose_dict,
                   show=False,
                   save=False,
                   show_lidar=True,
                   show_pred=False,
                   subject_token=None):
    
    points, cp_points = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    # if we only want top lidar then it's only the first index of points: lidar_points = points[0]

    # Lidar info and figure setup
    lidar_points = np.concatenate(points, axis=0)
    fig = plt.figure(frameon=False)
    DPI = fig.get_dpi()
    fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
    ax = fig.add_subplot(111, xticks=[], yticks=[])
    height = lidar_points[:,2]
    intensity = lidar_points[:,3]

    # lidar plotting, right now we are using the same color for ground and non ground points (red=gray)
    if show_lidar:
        gray = [153/255, 153/255, 153/255]
    else:
        gray = [1, 1, 1]
    red = gray
    ground_points = lidar_points[height<0.7,:] # meters threshold
    non_ground_points = lidar_points[height>0.7,:] # meters threshold
    ax.scatter(x = ground_points[:,0], y=ground_points[:,1], s = 0.01, c=np.tile(gray,(ground_points.shape[0],1)))
    ax.scatter(x = non_ground_points[:,0], y=non_ground_points[:,1], s = 0.01, c=np.tile(red,(non_ground_points.shape[0],1)))

    ### plot adjustments
    ax.set_xlim(-100,10)
    ax.set_ylim(-100,100)

    ax.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    
    ax.axis('on')
    
    if last_frame:
        keys_list = [key for key in predictions.keys()]
        # print(keys_list)
        frames = keys_list[seq_start_idx:seq_end_idx]
        points, cp_points = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        lidar_points = np.concatenate(points, axis=0)
        fig = plt.figure(frameon=False)
        DPI = fig.get_dpi()
        fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
        ax = fig.add_subplot(111, xticks=[], yticks=[])
        height = lidar_points[:,2]
        intensity = lidar_points[:,3]

        gray = [153/255, 153/255, 153/255]
        ground_points = lidar_points[height<0.7,:] #meters threshold
        non_ground_points = lidar_points[height>0.7,:] #meters threshold
        ax.scatter(x = ground_points[:,0], y=ground_points[:,1], s = 0.01, c=np.tile(gray,(ground_points.shape[0],1)))
        ax.scatter(x = non_ground_points[:,0], y=non_ground_points[:,1], s = 0.01, c=np.tile(gray,(non_ground_points.shape[0],1)))


        def transform_box_to_ego(box, rotation, translation):
            box_coord = np.array([box[0], box[1], box[2]])
            box_coord = np.dot(rotation, box_coord - translation)
            box[0], box[1], box[2] = box_coord[0], box_coord[1], box_coord[2]
            return box
            
        
        tracks = {}
        
        for i, token in enumerate(frames):
            # Get current frame tracking result and transformation
            cur_frame_result = gt[token]
            cur_frame_transform = ego_pose_dict[token]
            cur_frame_rotation, cur_frame_translation = get_total_transform(ego_pose_dict, i)
            cur_frame_rotation_matrix = rotation_matrix_to_euler_angles(cur_frame_rotation)
            
            # Get all tracking ids, box3d_lidar, label_preds, for the current frame
            tracking_ids = cur_frame_result['tracking_ids']
            box3d_lidar = cur_frame_result['box3d_lidar']
            label_preds = cur_frame_result['label_preds']

            for idx, t_id in enumerate(tracking_ids):
                box3d = box3d_lidar[idx]
                box3d = transform_box_to_ego(box3d, cur_frame_rotation, cur_frame_translation)
                if t_id not in tracks:
                    tracks[t_id] = [box3d]
                else:
                    tracks[t_id].append(box3d)
        
        for t_id, coords in tracks.items():
            for c in coords:
                x, y = c[0], c[1]
                ax.plot(x, y, 'o-', color='blue')

        ax.plot(0, 0, 'x', color='black')
        ### plot adjustments
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.axis('off')

        # active_tokens = "Tokens in this scene: \n\n" + "\n".join(gt_tracking_ids)
        # ax.text(-95, -95, active_tokens, fontsize = 7, color ="blue")
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        ax.axis('on')
        if show:
            plt.show()
        if save:
            print(f'saved to: {track_file}')
            cur_frame_rotation_matrix = [float(str(i)[:5]) for i in cur_frame_rotation_matrix]
            fig.suptitle(f"Scene: {token} \n rotation: {cur_frame_rotation_matrix} \n translation: {cur_frame_translation}", fontsize=20) 
            fig.savefig(track_file)
            plt.close('all')
        

def extract_frame_with_token(file, token):
    info_dir = os.path.dirname(file)
    lidar_path = os.path.join(info_dir, 'val', 'lidar', token)
    pkl_file = np.load(lidar_path, allow_pickle=True)
    scene_name = pkl_file['scene_name']
    frame_name = pkl_file['frame_name']
    frame_id = pkl_file['frame_id']
    lidars = pkl_file['lidars']
    prefix = 'segment-'
    suffix = '_with_camera_labels.tfrecord'
    scene_name = prefix + scene_name + suffix
    scene_path = os.path.join(info_dir, 'val', scene_name)
    return scene_path

def render_boxes(output_folder, 
                 file, 
                 predictions, 
                 token, 
                 frame_id, 
                 last_frame, 
                 seq_start_idx, 
                 seq_end_idx, 
                 gt, 
                 infos, 
                 idx,
                 ego_pose_dict, 
                 subject_token=None):
    '''
    output_folder: folder to save the visualizations
    file: ground truth file path
    predictions: dict, key is token, value is detection for that token
                detection is also a dict, keys: ['tracking_ids', 'box3d_lidar', 'scores', 'pose']
    token: current token for visualization (tracked up to this current token)
    frame_id: frame number for the current sequence (for example if token is seq_1_frame_2.pkl then frame_id is 2)
    last_frame: whether to visualize tracking (always True right now)
    seq_start_idx: starting index of the current sequence in
    seq_end_idx: end index of the current sequence
    gt: ground truth dict tracking that contains all detections from all frames of all sequences
    infos: ground truth detections (all)
    idx: idx in the entire dataset
    ego_pose_list: a list of transformations to get the detections back to the ego
    '''
    file = extract_frame_with_token(file, token)
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    # print(f'token: {token}')
    for indx, data in enumerate(dataset):
        if indx == frame_id:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # print(f'frame.context.name: {frame.context.name}')
            (range_images, camera_projections, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
            out_file1 = output_folder+os.path.basename(file)
            create_dir(out_file1)
            out_file = out_file1+"/"+str(indx).zfill(6)+".png"
            track_file = out_file1+"/"+str(indx).zfill(6)+"_track"+".png"
            lidar_top_view(predictions, frame, range_images, camera_projections, range_image_top_pose, token, out_file, track_file, frame_id, last_frame, seq_start_idx, seq_end_idx, gt, infos, idx, ego_pose_dict, save=True, subject_token=subject_token)
    # print(file)