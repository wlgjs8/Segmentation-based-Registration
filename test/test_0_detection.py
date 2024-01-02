import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import numpy as np

from model.detection_model import HourGlass3D
from dataloader.detection_dataset import get_detection_dataloader
from transform.detection_transform import train_detection_transform, test_detection_transform

from utils import hadamard_product, save_detection_result
import eval_utils

model = HourGlass3D(
    nStack = 2,
    nBlockCount = 4,
    nResidualEachBlock = 1,
    nMidChannels = 128,
    nChannels = 128,
    nJointCount = 128,
    bUseBn = True,
)

MODEL_WEIGHT_PATH = './checkpoints/detection_best_model.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

model = model.cuda()

_, test_dataloader = get_detection_dataloader(train_detection_transform, test_detection_transform)

box_iou_list = []

with torch.no_grad():
    for idx, data in enumerate(test_dataloader):

        person_id = data['person_id']
        flag = data['flag']
        cls = data['cls']

        resize_image = data['resize_image']
        resize_bbox = data['resize_bbox']
        resize_center = data['resize_center']
        resize_heatmap = data['resize_heatmap']

        _ , output, offset = model(resize_image)

        gt_image = resize_image.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_heatmap = resize_heatmap.squeeze(0).detach().cpu().numpy()
        gt_center = resize_center.squeeze(0).detach().cpu().numpy()
        gt_bbox = resize_bbox.squeeze(0).detach().cpu().numpy()
        gt_cls = cls.squeeze(0).detach().cpu().numpy()

        pred_heatmap = output.squeeze(0).detach().cpu().numpy()
        pred_center = hadamard_product(output.squeeze(0)).detach().cpu().numpy()
        pred_offset = offset.squeeze(0).detach().cpu().numpy()

        pred_coord1 = pred_center - pred_offset/2
        pred_coord2 = pred_center + pred_offset/2
        pred_bbox = np.concatenate((pred_coord1, pred_coord2), axis=1)
    
        if idx == 0:
            save_detection_result(gt_image, 
                          gt_heatmap, gt_center, gt_bbox, gt_cls,
                          pred_heatmap, pred_center, pred_bbox)

        box_iou = eval_utils.box_iou(pred_bbox[gt_cls==1], gt_bbox)
        box_iou_list.append(box_iou)
        # print('box_iou : ', box_iou)
        # print()

print('Avg box iou : ', sum(box_iou_list)/len(box_iou_list))