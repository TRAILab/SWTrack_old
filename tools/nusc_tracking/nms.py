import numpy as np

def filter_detections(detections, score_thresh=0.0001, iou_thresh=0.1):
    """Filter detections based on score and extra NMS."""
    # Filter out low score detections
    score_cond = np.array([det['detection_score'] > score_thresh
                    for det in detections])
    # Filter out detections with same class and location (NMS)
    class_ids = np.array([det['label_preds'] for det in detections])
    same_class_mat = np.equal(
        class_ids.reshape(-1, 1),
        class_ids.reshape(1, -1))
    pos_x = np.array([det['translation'][0] for det in detections])
    pos_y = np.array([det['translation'][1] for det in detections])
    size_w = np.array([det['size'][0] for det in detections])
    size_l = np.array([det['size'][1] for det in detections])
    rot = np.array([det['rotation'][2] for det in detections])
    ind_remove = np.zeros(len(detections), dtype=np.bool)
    max_iou = np.zeros(len(detections))
    for i in range(len(detections)):
        if ind_remove[i]:
            continue
        iou_vec = np.zeros(len(detections))
        for j in range(i+1, len(detections)):
            dist = np.sqrt((pos_x[i] -
                            pos_x[j])**2 +
                            (pos_y[i] -
                                pos_y[j])**2)
            max_dim = np.max([size_w[i],
                                size_l[i],
                                size_w[j],
                                size_l[j]])
            if same_class_mat[i, j] and dist < max_dim and not ind_remove[j]:
                iou_vec[j] = iou2d(
                    [pos_x[i],
                    pos_y[i],
                    size_w[i],
                    size_l[i],
                    rot[i]],
                    [pos_x[j],
                    pos_y[j],
                    size_w[j],
                    size_l[j],
                    rot[j]])
        ind_remove += (iou_vec > iou_thresh)
        max_iou[i] = np.max(iou_vec)
    nms_cond = np.logical_not(ind_remove)
    # Filter detections
    x_cond = np.logical_and(score_cond, nms_cond)
    detections = [det for det, cond in zip(detections, x_cond) if cond]
    return detections


def iou2d(box_a, box_b):
    """
    Compute 2D bounding box IoU.
    
    Args:
        box_a (list[float]): Box A with format [x, y, w, l, r].
        box_b (list[float]): Box B with format [x, y, w, l, r].
    """
    def box2corners2d(bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox[0], bbox[1]])
        cos, sin = np.cos(bbox[4]), np.sin(bbox[4])
        pc0 = np.array([bbox[0] + cos * bbox[3] / 2 + sin * bbox[2] / 2,
                        bbox[1] + sin * bbox[3] / 2 - cos * bbox[2] / 2])
        pc1 = np.array([bbox[0] + cos * bbox[3] / 2 - sin * bbox[2] / 2,
                        bbox[1] + sin * bbox[3] / 2 + cos * bbox[2] / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    from shapely.geometry import Polygon
    boxa_corners = np.array(box2corners2d(box_a))
    boxb_corners = np.array(box2corners2d(box_b))
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou
