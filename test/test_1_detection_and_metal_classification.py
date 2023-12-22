import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
# sys.path.append('../')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from draw import draw_utils

from config import (
    TRAINING_EPOCH, NUM_CLASSES, RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH,
    LOWER_TOOTH_NUM, UPPER_TOOTH_NUM
)
# from hourglass import HourGlass3D
from model.metal_concat_crop_no import HourGlass3D
# from model.metal_concat_crop_all import HourGlass3D

# from dataloader.score_dataset import get_detection_dataloader
from dataloader.detection_dataset_origin import get_inference_dataloader

from utils import save_result, save_points2, generate_gaussian_heatmap_tensor

import utils
from utils import save_file, hadamard_product, _tranpose_and_gather_feature, get_maximum_point_tensor
import numpy as np

writer = SummaryWriter()

model = HourGlass3D(
    nStack = 2,
    nBlockCount = 4,
    nResidualEachBlock = 1,
    nMidChannels = 128,
    nChannels = 128,
    nJointCount = 128,
    bUseBn = True,
)

# MODEL_WEIGHT_PATH = './checkpoints(#m2)/epoch30_iou_valLoss0.00013813463519298959.pth'
# MODEL_WEIGHT_PATH = './checkpoints/epoch41_acc0.8943661908847451.pth'
# MODEL_WEIGHT_PATH = './checkpoints/epoch88_acc0.8194444387538581.pth'

MODEL_WEIGHT_PATH = './checkpoints/epoch70_acc0.7151515108172636.pth'


model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model = model.cuda()
model.eval()

inference_dataloader = get_inference_dataloader()
# SAVE_DIR = os.path.join('C:/Users/JeeheonKim/source/ct/pose/ct_model_box_image')
SAVE_DIR = os.path.join('C:/Users/JeeheonKim/source/ct/pose/ct_model_box')

with torch.no_grad():

    for idx, data in enumerate(inference_dataloader):
        person_id = data['person_id']
        flag = data['flag']
        print('person_id : {}, flag : {}'.format(person_id[0], flag[0]))

        resize_image = data['resize_image']
        inter_output, output, offset, pred_center, score_output = model(resize_image)

        SAVE_PERSON_DIR = os.path.join(SAVE_DIR, person_id[0])
        
        score_output = score_output.squeeze(0).detach().cpu().numpy()
        pred_metal_list = np.argmax(score_output, 1)

        pred_center = hadamard_product(output.squeeze(0))
        norm_pred_center = pred_center.squeeze(0)

        norm_offset = offset.squeeze(0)
        coor1 = norm_pred_center - norm_offset/2
        coor2 = norm_pred_center + norm_offset/2
        pred_box = torch.cat([coor1, coor2], dim=1)
        pred_box = pred_box.detach().cpu().numpy()

        draw_pred_box = pred_box * 1.75

        draw_utils.draw_metal_boxes(draw_pred_box, pred_metal_list==0, person_id[0], flag[0], category='no', margin=False, shape=(112, 224, 224))
        draw_utils.draw_metal_boxes(draw_pred_box, pred_metal_list==1, person_id[0], flag[0], category='normal', margin=False, shape=(112, 224, 224))
        draw_utils.draw_metal_boxes(draw_pred_box, pred_metal_list==2, person_id[0], flag[0], category='metal', margin=False, shape=(112, 224, 224))

        print('pred_box : ', pred_box.shape)
        print('pred_metal_list : ', pred_metal_list)
        # pred_box = pred_box[pred_metal_list==1]

        if flag[0] == 'upper':
            TOOTH_NUM = UPPER_TOOTH_NUM
        elif flag[0] == 'lower':
            TOOTH_NUM = LOWER_TOOTH_NUM
        else:
            print('fuck!')

        no_metal_tooth_list = TOOTH_NUM
        no_metal_tooth_list = np.array(no_metal_tooth_list)
        # print('no_metal_tooth_list : ', no_metal_tooth_list)
        # print('pred_metal_list : ', pred_metal_list)
        no_metal_tooth_list = no_metal_tooth_list[pred_metal_list==1]
        # print('score_output : ',score_output)

        box_dict = dict()
        for idx in range(len(no_metal_tooth_list)):
            tooth_idx = no_metal_tooth_list[idx]
            box_dict[tooth_idx] = pred_box[idx].tolist()

        save_path = os.path.join(SAVE_PERSON_DIR, 'box_pred_{}_no.json'.format(flag[0]))
        with open(save_path, 'w') as jr:
            json.dump(box_dict, jr, indent=4)
        jr.close()
