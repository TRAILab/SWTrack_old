import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import os
from utils import *
import glob
from joblib import Parallel, delayed
import multiprocessing
from progressbar import ProgressBar
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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

def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    # breakpoint()
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 
    # breakpoint()
    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)
def lidar_top_view(predictions, frame, range_images, camera_projections, range_image_top_pose, token, out_file, track_file, frame_id, last_frame, seq_start_idx, seq_end_idx, gt, show=False, save=False, show_lidar=True):
    points, cp_points = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # merged lidars
    lidar_points = np.concatenate(points, axis=0)
    fig = plt.figure(frameon=False)
    DPI = fig.get_dpi()
    fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
    ax = fig.add_subplot(111, xticks=[], yticks=[])
    height = lidar_points[:,2]
    intensity = lidar_points[:,3]
    tracking_labels = {
        0: 'car',
        1: 'pedestrian',
        2: 'cyclist'
    }
    tracking_colors = {
        'car': 'blue',
        'pedestrian': 'brown',
        'cyclist': 'cyan'
    }
    if token not in list(gt.keys()):
        print('Token is not in ground truth keys.')
        breakpoint()
    ######## style 1: combined height and intensity map ########
    # height = np.interp(height, (height.min(), height.max()), (0, 1))
    # # height = np.clip(height, 0, 1)
    # height = np.expand_dims(height, axis=1)
    # intensity = np.expand_dims(intensity, axis=1)
    # zeros = np.zeros_like(height)
    # colors = np.hstack((zeros, height, intensity))
    # ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=colors)

    ######## style 2: using height to visuzalize ground and obstacles (precog paper style) ########
    if show_lidar:
        gray = [153/255, 153/255, 153/255]
    else:
        gray = [1, 1, 1]
    red = gray # [228/255, 27/255, 28/255]
    ground_points = lidar_points[height<0.7,:] #meters threshold
    non_ground_points = lidar_points[height>0.7,:] #meters threshold
    ax.scatter(x = ground_points[:,0], y=ground_points[:,1], s = 0.01, c=np.tile(gray,(ground_points.shape[0],1)))
    ax.scatter(x = non_ground_points[:,0], y=non_ground_points[:,1], s = 0.01, c=np.tile(red,(non_ground_points.shape[0],1)))
    # breakpoint()
    # res = predictions['seq_0_frame_0.pkl']
    res = predictions[token]
    box3d = res['box3d_lidar']
    pose = res['pose']
    labels = res['label_preds']
    num_boxes = box3d.shape[0]
    box3d[:, [3, 4]] = box3d[:, [4, 3]]

    ## do the same for gt boxes:
    gt_res = gt[token]
    gt_box3d = gt_res['box3d_lidar']
    gt_labels =  gt_res['label_preds']
    num_gt_boxes = gt_box3d.shape[0]
    # breakpoint()
    gt_box3d[:, [3, 4]] = gt_box3d[:, [4, 3]]
    ## end gt changes

    # for loop for model detections
    for i in range(num_boxes):
        box = box3d[i]
        # yaw = box[-1]#  + np.pi / 2 #np.arctan2(pose[1, 0], pose[0,0]) + np.pi/2
        # print(f'center: {box[0]}')
        yaw = box[-1] - np.pi / 2
        deg = np.degrees(yaw)
        # deg = 0
        transform = mpl.transforms.Affine2D().rotate_deg_around(box[0], box[1], deg) + ax.transData
        bot_x, bot_y = box[0] - box[3]/2, box[1] - box[4]/2
        rectangle = patches.Rectangle(
            [bot_x, bot_y],  # (x, y) coordinates of the bottom left corner   
            box[3], # Width of the rectangle
            box[4],  # Height of the rectangle,
            linewidth=2,
            edgecolor=tracking_colors[tracking_labels[labels[i].item()]], #'blue',   # Color of the rectangle's border
            facecolor='none'    # Transparent inside the rectangle
        )
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        ax.plot(box[0], box[1], 'x', color='black')
    # for loop for ground truth detections
    for i in range(num_gt_boxes):
        gt_box = gt_box3d[i]
        yaw = gt_box[-1] - np.pi / 2
        deg = np.degrees(yaw) # + np.pi / 2
        transform = mpl.transforms.Affine2D().rotate_deg_around(gt_box[0], gt_box[1], deg) + ax.transData
        gt_bot_x, gt_bot_y = gt_box[0] - gt_box[3]/2, gt_box[1] - gt_box[4]/2
        gt_rectangle = patches.Rectangle(
            [gt_bot_x, gt_bot_y],
            gt_box[3],
            gt_box[4],
            linewidth=2,
            edgecolor='black', # doing all black rectangles for ground truths for now
            facecolor='none'
        )
        gt_rectangle.set_transform(transform)
        ax.add_patch(gt_rectangle)
        ax.plot(gt_box[0], gt_box[1], '*', color='green') # red centers for ground truth centers 

    ax.plot(0, 0, 'x', color='black')
    ### plot adjustments
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    # breakpoint()
    handles = [
        Line2D([0], [0], label='annotations', color='black'),
        Line2D([0], [0], label='car', color=tracking_colors['car']),
        Line2D([0], [0], label='pedestrian', color=tracking_colors['pedestrian']),
        Line2D([0], [0], label='cyclist', color=tracking_colors['cyclist'])
    ]
    ax.legend(handles=handles, fontsize='large', loc='upper right')
    ax.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    ax.axis('on')
    if show:
        plt.show()
    if save:
        print(f'saved to: {out_file}')
        fig.savefig(out_file)
        plt.close('all')
    if last_frame:
        active_tracking_ids = res['tracking_ids']
        gt_active_tracking_ids = gt_res['tracking_ids']
        # print('rendering the tracking results for this frame now')
        # breakpoint()
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
        ######## style 1: combined height and intensity map ########
        # height = np.interp(height, (height.min(), height.max()), (0, 1))
        # # height = np.clip(height, 0, 1)
        # height = np.expand_dims(height, axis=1)
        # intensity = np.expand_dims(intensity, axis=1)
        # zeros = np.zeros_like(height)
        # colors = np.hstack((zeros, height, intensity))
        # ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=colors)

        ######## style 2: using height to visuzalize ground and obstacles (precog paper style) ########
        if show_lidar:
            gray = [153/255, 153/255, 153/255]
        else:
            gray = [1, 1, 1]
        red = gray
        ground_points = lidar_points[height<0.7,:] #meters threshold
        non_ground_points = lidar_points[height>0.7,:] #meters threshold
        ax.scatter(x = ground_points[:,0], y=ground_points[:,1], s = 0.01, c=np.tile(gray,(ground_points.shape[0],1)))
        ax.scatter(x = non_ground_points[:,0], y=non_ground_points[:,1], s = 0.01, c=np.tile(red,(non_ground_points.shape[0],1)))
        # tracking for loop for detections
        for j in range(len(frames)):
            token = frames[j]
            res = predictions[token]
            box3d = res['box3d_lidar']
            pose = res['pose']
            labels = res['label_preds']
            num_boxes = box3d.shape[0]
            box3d[:, [3, 4]] = box3d[:, [4, 3]]
            tracking_ids = res['tracking_ids']
            if j < len(frames) - 1:
                next_token = frames[j+1]
                next_res = predictions[next_token]
                next_box3d = next_res['box3d_lidar']
                next_pose = next_res['pose']
                next_num_boxes = next_box3d.shape[0]
                next_box3d[:, [3, 4]] = next_box3d[:, [4, 3]]
                next_tracking_ids = next_res['tracking_ids']
            for i in range(len(tracking_ids)):
                if tracking_ids[i] not in active_tracking_ids:
                    print('not active, skipped!')
                    continue
                box = box3d[i]
                bot_x, bot_y = box[0] - box[3]/2, box[1] - box[4]/2
                box = box3d[i]
                yaw = box[-1] - np.pi/2 #- np.arctan2(pose[1, 0], pose[0,0]) + np.pi/2
                # print(f'center: {box[0]}')
                # yaw = box[-1] - np.pi / 2
                deg = np.degrees(yaw)
                # deg = 0
                transform = mpl.transforms.Affine2D().rotate_deg_around(box[0], box[1], deg) + ax.transData
                need_box = False
                if j == len(frames)-1:
                    rectangle = patches.Rectangle(
                        [bot_x, bot_y],  # (x, y) coordinates of the bottom left corner   
                        box[3], # Width of the rectangle
                        box[4],  # Height of the rectangle,
                        linewidth=2,
                        edgecolor=tracking_colors[tracking_labels[labels[i].item()]],   # Color of the rectangle's border
                        facecolor='none'    # Transparent inside the rectangle
                    )
                    rectangle.set_transform(transform)
                    ax.add_patch(rectangle)
                    next_tracking_ids = np.array([])
                    # need_box = True
                    # print('plotted box since j == len(frames-1)')
                if tracking_ids[i] not in next_tracking_ids:
                    rectangle = patches.Rectangle(
                        [bot_x, bot_y],  # (x, y) coordinates of the bottom left corner   
                        box[3], # Width of the rectangle
                        box[4],  # Height of the rectangle,
                        linewidth=2,
                        edgecolor=tracking_colors[tracking_labels[labels[i].item()]],   # Color of the rectangle's border
                        facecolor='none'    # Transparent inside the rectangle
                    )
                    rectangle.set_transform(transform)
                    ax.add_patch(rectangle)
                    # print('plotted box since tracking ended')
                elif j != len(frames)-1 and tracking_ids[i] in next_tracking_ids and tracking_ids[i] in active_tracking_ids:
                    print(f'next frames also has {tracking_ids[i]}')
                    # breakpoint()
                    next_tracking_ids_list = next_tracking_ids.tolist()
                    next_frame_track_id = next_tracking_ids_list.index(tracking_ids[i])
                    next_box = next_box3d[next_frame_track_id]
                    print(f'Drawing line between {(box[0], box[1])} and {(next_box[0], next_box[1])}')
                    dx, dy =  next_box[0]-box[0], next_box[1]-box[1]
                    ax.arrow(box[0].item(), box[1].item(), dx.item(), dy.item(), linewidth=1, color='purple')
                    # breakpoint()
                ax.plot(box[0], box[1], 'x', color='black')
        
        # tracking for loop for ground truth
        for j in range(len(frames)):
            token = frames[j]
            res = gt[token]
            box3d = res['box3d_lidar']
            # pose = res['pose']
            labels = res['label_preds']
            num_boxes = box3d.shape[0]
            box3d[:, [3, 4]] = box3d[:, [4, 3]]
            tracking_ids = res['tracking_ids']
            if j < len(frames) - 1:
                next_token = frames[j+1]
                next_res = gt[next_token]
                next_box3d = next_res['box3d_lidar']
                # next_pose = next_res['pose']
                next_num_boxes = next_box3d.shape[0]
                next_box3d[:, [3, 4]] = next_box3d[:, [4, 3]]
                next_tracking_ids = next_res['tracking_ids']
            # breakpoint()
            for i in range(len(tracking_ids)):
                if tracking_ids[i] not in gt_active_tracking_ids:
                    print('not active, skipped!')
                    continue
                box = box3d[i]
                bot_x, bot_y = box[0] - box[3]/2, box[1] - box[4]/2
                box = box3d[i]
                yaw = box[-1] - np.pi/2 #- np.arctan2(pose[1, 0], pose[0,0]) + np.pi/2
                # print(f'center: {box[0]}')
                # yaw = box[-1] - np.pi / 2
                deg = np.degrees(yaw)
                # deg = 0
                transform = mpl.transforms.Affine2D().rotate_deg_around(box[0], box[1], deg) + ax.transData
                # need_box = False
                if j == len(frames)-1:
                    rectangle = patches.Rectangle(
                        [bot_x, bot_y],  # (x, y) coordinates of the bottom left corner   
                        box[3], # Width of the rectangle
                        box[4],  # Height of the rectangle,
                        linewidth=2,
                        edgecolor='black',   # Color of the rectangle's border
                        facecolor='none'    # Transparent inside the rectangle
                    )
                    rectangle.set_transform(transform)
                    ax.add_patch(rectangle)
                    next_tracking_ids = np.array([])
                    # need_box = True
                    # print('plotted box since j == len(frames-1)')
                if tracking_ids[i] not in next_tracking_ids:
                    rectangle = patches.Rectangle(
                        [bot_x, bot_y],  # (x, y) coordinates of the bottom left corner   
                        box[3], # Width of the rectangle
                        box[4],  # Height of the rectangle,
                        linewidth=2,
                        edgecolor='black',   # Color of the rectangle's border
                        facecolor='none'    # Transparent inside the rectangle
                    )
                    rectangle.set_transform(transform)
                    ax.add_patch(rectangle)
                    # print('plotted box since tracking ended')
                elif j != len(frames)-1 and tracking_ids[i] in next_tracking_ids and tracking_ids[i] in gt_active_tracking_ids:
                    print(f'next frames also has {tracking_ids[i]}')
                    # breakpoint()
                    next_tracking_ids_list = next_tracking_ids # .tolist()
                    # breakpoint()
                    next_frame_track_id = next_tracking_ids_list.index(tracking_ids[i])
                    next_box = next_box3d[next_frame_track_id]
                    print(f'Drawing line between {(box[0], box[1])} and {(next_box[0], next_box[1])}')
                    dx, dy =  next_box[0]-box[0], next_box[1]-box[1]
                    ax.arrow(box[0].item(), box[1].item(), dx.item(), dy.item(), linewidth=1, color='green')
                    # breakpoint()
                ax.plot(box[0], box[1], '*', color='red')

        ax.plot(0, 0, 'x', color='black')
        ### plot adjustments
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.axis('off')
        handles = [
            Line2D([0], [0], label='annotations', color='black'),
            Line2D([0], [0], label='car', color=tracking_colors['car']),
            Line2D([0], [0], label='pedestrian', color=tracking_colors['pedestrian']),
            Line2D([0], [0], label='cyclist', color=tracking_colors['cyclist'])
        ]
        ax.legend(handles=handles, fontsize='large', loc='upper right')
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        ax.axis('on')
        if show:
            plt.show()
        if save:
            print(f'saved to: {track_file}')
            fig.savefig(track_file)
            plt.close('all')
        

def extract_frame_with_token(file, token):
    # breakpoint()
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
    # breakpoint()
    return scene_path

def render_boxes(output_folder, file, predictions, token, frame_id, last_frame, seq_start_idx, seq_end_idx, gt, top_view_lidar_image=True, ego_vehicle_motion=False):
    # create_dir(outfolder)
    file = extract_frame_with_token(file, token)
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    print(f'file: {file}')
    print(f'token: {token}')
    # breakpoint()
    # read all frames
    for indx, data in enumerate(dataset):
        if indx == frame_id:
            frame_data = data.numpy()
            frame = open_dataset.Frame()
            # breakpoint()
            # breakpoint()
            # frame_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            frame.ParseFromString(bytearray(data.numpy()))
            print(f'frame.context.name: {frame.context.name}')
            (range_images, camera_projections, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
            # breakpoint()
            # # 1.camera images
            # if camera_images:
            # 	visualize_cameras(frame, range_images, camera_projections, range_image_top_pose, outfolder, indx, save=True)
            # top view lidar image
            # breakpoint()
            if top_view_lidar_image:
                # out_file1 = outfolder+"/lidar_top"
                # out_file1 = output_folder+"lidar_top"+'/'+os.path.basename(file)
                out_file1 = output_folder+os.path.basename(file)
                # breakpoint()
                create_dir(out_file1)
                out_file = out_file1+"/"+str(indx).zfill(6)+".png"
                track_file = out_file1+"/"+str(indx).zfill(6)+"_track"+".png"
                # breakpoint()
                #if not os.path.isfile(out_file):
                lidar_top_view(predictions, frame, range_images, camera_projections, range_image_top_pose, token, out_file, track_file, frame_id, last_frame, seq_start_idx, seq_end_idx, gt, save=True)
        

    print(file)
 
# if __name__ == '__main__':
#     tf_path = '/home/robert/Desktop/trail/CenterPoint/data/Waymo/val/segment-17065833287841703_2980_000_3000_000_with_camera_labels.tfrecord'
#     render_boxes(tf_path)