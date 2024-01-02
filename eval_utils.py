import numpy as np
from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)


def box_iou(boxes1, boxes2):
    ious = []
    for i in range(len(boxes1)):
        box1 = boxes1[i]
        box2 = boxes2[i]

        box1_size = np.maximum(box1[3:] - box1[:3], 0)
        box2_size = np.maximum(box2[3:] - box2[:3], 0)

        box1_volume = np.prod(box1_size)
        box2_volume = np.prod(box2_size)

        intersection_size = np.maximum(np.minimum(box1[3:], box2[3:]) - np.maximum(box1[:3], box2[:3]), 0)
        intersection_volume = np.prod(intersection_size)

        iou = intersection_volume / (box1_volume + box2_volume - intersection_volume + 1e-6)
        ious.append(iou)

    return sum(ious)/len(ious)


def box_oir(pred_bboxes, gt_bboxes, gt_maskes, margin=False):
    '''
    박스가 얼마나 치아를 포함하는지 나타내는 Organ In Region(OIR) 성능 측정 코드, 
    '''
    oirs = []
    for i in range(len(pred_bboxes)):
        pred_bbox = pred_bboxes[i].astype(int)
        gt_bbox = gt_bboxes[i].astype(int)
        gt_mask = gt_maskes[i]

        if margin==True:
            pred_bbox = add_margin(pred_bbox)

        pred_bbox = fill_box(pred_bbox)
        gt_bbox = fill_box(gt_bbox)

        overlap = pred_bbox * gt_mask
        union = gt_bbox * gt_mask

        oir = overlap.sum() / union.sum()    
        oirs.append(oir)

    return sum(oirs)/len(oirs)


def fill_box(box):
    '''
    박스를 Slicer 에서 보기 위해선, 
    점 2개가 아니라 모서리가 채워진 박스이어야 하기에 점 2개를 통해 모서리 부분을 1로 색칠함.
    '''
    ds = min(box[0], box[3])
    de = max(box[0], box[3])

    hs = min(box[1], box[4])
    he = max(box[1], box[4])
    
    ws = min(box[2], box[5])
    we = max(box[2], box[5])

    if ds < 0:
        ds = 0
    if hs < 0:
        hs = 0
    if ws < 0:
        ws = 0
    if de >= RESIZE_DEPTH:
        de = RESIZE_DEPTH-1
    if he >= RESIZE_HEIGHT:
        he = RESIZE_HEIGHT-1
    if we >= RESIZE_WIDTH:
        we = RESIZE_WIDTH-1

    area = np.zeros((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))

    for didx in range(ds, de+1):
        for hidx in range(hs, he+1):
            for widx in range(ws, we+1):
                area[didx][hidx][widx] = 1
    
    return area


def add_margin(box, margin=2):
    '''
    OIR 성능 측정시에 Box 에 Margin 을 더하는 함수.
    '''
    ds = min(box[0], box[3])
    de = max(box[0], box[3])

    hs = min(box[1], box[4])
    he = max(box[1], box[4])
    
    ws = min(box[2], box[5])
    we = max(box[2], box[5])

    margin_bbox = [
        ds - margin, hs - margin, ws - margin,
        de + margin, he + margin, we + margin,
    ]

    if margin_bbox[0] < 0:
        margin_bbox[0] = 0
    if margin_bbox[1] < 0:
        margin_bbox[1] = 0
    if margin_bbox[2] < 0:
        margin_bbox[2] = 0
    if margin_bbox[3] >= RESIZE_DEPTH:
        margin_bbox[3] = RESIZE_DEPTH-1
    if margin_bbox[4] >= RESIZE_HEIGHT:
        margin_bbox[4] = RESIZE_HEIGHT-1
    if margin_bbox[5] >= RESIZE_WIDTH:
        margin_bbox[5] = RESIZE_WIDTH-1

    return np.array(margin_bbox)
