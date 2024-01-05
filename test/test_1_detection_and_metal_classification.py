import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import numpy as np
import torch

from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH, LOWER_TOOTH_NUM, UPPER_TOOTH_NUM
)
from model.detection_and_metal_classification_model import HourGlass3D
from dataloader.detection_and_metal_classification_dataset import get_inference_dataloader

from utils import hadamard_product, save_detection_and_metal_classification_result


model = HourGlass3D(
    nStack = 2,
    nBlockCount = 4,
    nResidualEachBlock = 1,
    nMidChannels = 128,
    nChannels = 128,
    nJointCount = 128,
    bUseBn = True,
)

MODEL_WEIGHT_PATH = './checkpoints/epoch1_valLoss0.0009223660860126044.pth'
# MODEL_WEIGHT_PATH = './checkpoints/detection_and_metal_classification_best_model.pth'

model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model = model.cuda()
model.eval()

inference_dataloader = get_inference_dataloader()
SAVE_DIR = os.path.join('C:/Users/JeeheonKim/source/Segmentation based Registration/results/detection_and_metal_classification')

with torch.no_grad():
    for idx, data in enumerate(inference_dataloader):

        person_id = data['person_id']
        flag = data['flag']
        cls = data['cls']

        resize_image = data['resize_image']
        resize_bbox = data['resize_bbox']
        resize_center = data['resize_center']
        resize_heatmap = data['resize_heatmap']

        _, output, offset, score_output = model(resize_image)

        
        score_output = score_output.squeeze(0).detach().cpu().numpy()
        pred_metal_list = np.argmax(score_output, 1)

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


        # if idx == 0:
        #     save_detection_and_metal_classification_result(gt_image, 
        #                   gt_heatmap, gt_center, gt_bbox, gt_cls, pred_metal_list,
        #                   pred_heatmap, pred_center, pred_bbox)
            

        SAVE_PERSON_DIR = os.path.join(SAVE_DIR, person_id[0])
        if not os.path.exists(SAVE_PERSON_DIR):
            os.mkdir(SAVE_PERSON_DIR)

        if flag[0] == 'upper':
            TOOTH_NUM = UPPER_TOOTH_NUM
        elif flag[0] == 'lower':
            TOOTH_NUM = LOWER_TOOTH_NUM

        no_metal_tooth_list = TOOTH_NUM
        no_metal_tooth_list = np.array(no_metal_tooth_list)
        no_metal_tooth_list = no_metal_tooth_list[pred_metal_list==0]

        # box_dict = dict()
        # for idx in range(len(no_metal_tooth_list)):
        #     tooth_idx = no_metal_tooth_list[idx]
        #     box_dict[tooth_idx] = pred_box[idx].tolist()

        # save_path = os.path.join(SAVE_PERSON_DIR, 'box_pred_{}_no.json'.format(flag[0]))
        # with open(save_path, 'w') as jr:
        #     json.dump(box_dict, jr, indent=4)
        # jr.close()
