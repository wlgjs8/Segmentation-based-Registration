import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch

from model.detection_model import HourGlass3D
from dataloader.detection_dataset import get_detection_dataloader
from transform.detection_transform import train_detection_transform, test_detection_transform


model = HourGlass3D(
    nStack = 2,
    nBlockCount = 4,
    nResidualEachBlock = 1,
    nMidChannels = 128,
    nChannels = 128,
    nJointCount = 128,
    bUseBn = True,
)

# MODEL_WEIGHT_PATH = './checkpoints/epoch11_valLoss0.006702427752315998.pth'
# model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model = model.cuda()

_, test_dataloader = get_detection_dataloader(train_detection_transform, test_detection_transform)



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

        # pred_heatmap = output[0, (cls==1)[0], :, :, :].unsqueeze(0)
        
        # temp_image = resize_image
        # temp_inter_target = None
        # temp_target = output
        
        # temp_gt_cls = cls
        # temp_gt_heatmap = resize_heatmap
        # temp_gt_center = resize_center
        # temp_gt_half_heatmap = resize_half_heatmap
        # temp_person_id = person_id[0]
        # temp_flag = flag[0]
        # temp_reg_points = reg_output

        # save_points(
        #     temp_image, temp_inter_target, temp_target, 
        #     temp_gt_cls, temp_gt_heatmap, temp_gt_center,
        #     temp_reg_points, temp_person_id, temp_flag
        # )