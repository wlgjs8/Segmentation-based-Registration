import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import (
    TRAINING_EPOCH, RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH, DR_EPOCH, METAL_EPOCH
)
from model.detection_and_metal_classification_model import HourGlass3D
from dataloader.detection_and_metal_classification_dataset import get_detection_dataloader
from transform.detection_and_metal_classification_transform import train_detection_transform, test_detection_transform
from loss.gaussian_disentangle import GDLoss
from loss.distance_regularization import DRLoss
from loss.losses import FocalLoss

from utils import hadamard_product


torch.manual_seed(123)

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

model = model.cuda()


train_dataloader, test_dataloader = get_detection_dataloader(train_detection_transform, test_detection_transform)
print('train_dataloader : ', len(train_dataloader))
print('test_dataloader : ', len(test_dataloader))

### Heatmap 에 대한 Loss 
### FocalLoss / Gaussian Disentangle Loss
hm_criterion = FocalLoss()
gd_criterion = GDLoss()
### Box 랑 치아의 중심점, Center 에 대한 Loss
box_criterion = nn.MSELoss()
pt_criterion = nn.MSELoss()
### 치아의 중심점 간의 Distance Regularization Loss
dr_criterion = DRLoss()
### 개별 치아에 대한 메탈인지 정상인지 분류하는 Loss
metal_score_criterion = nn.BCELoss()

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [100, 120, 360], gamma=0.1, last_epoch=-1)

min_valid_loss = math.inf
min_eval_err = math.inf
max_eval_iou = -1.

METAL_WEIGHT = 0.001

for epoch in range(TRAINING_EPOCH):
    train_losses = 0.0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    model.train()

    print()
    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        person_id = data['person_id']
        flag = data['flag']
        cls = data['cls']

        resize_image = data['resize_image']
        resize_bbox = data['resize_bbox'].type(torch.float32)
        resize_center = data['resize_center'].type(torch.float32)
        resize_heatmap = data['resize_heatmap']
        resize_half_heatmap = data['resize_half_heatmap']
        gt_metal_score = data['metal_score'].squeeze(0).type(torch.float32)
        
        
        ### 중심점 좌표를 0~1 로 정규화 한게 norm_resize_center
        norm_resize_center = resize_center.squeeze(0)
        norm_resize_center[:,0] = norm_resize_center[:,0] / RESIZE_DEPTH
        norm_resize_center[:,1] = norm_resize_center[:,1] / RESIZE_HEIGHT
        norm_resize_center[:,2] = norm_resize_center[:,2] / RESIZE_WIDTH
        norm_resize_center = norm_resize_center.unsqueeze(0)

        ### 치아 박스 좌표를 0~1 로 정규화 한게 norm_resize_bbox
        norm_resize_bbox = resize_bbox.squeeze(0)
        norm_resize_bbox[:,0] = norm_resize_bbox[:,0] / RESIZE_DEPTH
        norm_resize_bbox[:,1] = norm_resize_bbox[:,1] / RESIZE_HEIGHT
        norm_resize_bbox[:,2] = norm_resize_bbox[:,2] / RESIZE_WIDTH
        norm_resize_bbox[:,3] = norm_resize_bbox[:,3] / RESIZE_DEPTH
        norm_resize_bbox[:,4] = norm_resize_bbox[:,4] / RESIZE_HEIGHT
        norm_resize_bbox[:,5] = norm_resize_bbox[:,5] / RESIZE_WIDTH
        norm_resize_bbox = norm_resize_bbox.unsqueeze(0)

        ### 중간 heatmap output / 최종 heatmap output / 박스 중심점 기준 offset / 개별 치아에 대한 분류 e.g. Normal 치아 [1,0], Metal 치아 [0,1]
        inter_output, output, offset, score_output = model(resize_image)

        ### cls 는 Token 으로 생각해주면 됨. cls 의 shape은 [1, 16]이고, 이는 치아에 대해 GT 가 있는 애들만 학습하기 위해서 사전에 걸러내는 과정으로 사용됨.
        ### cls==1 이 True 면, GT가 있는 치아임.
        pred_heatmap = output[0, (cls==1)[0], :, :, :].unsqueeze(0)
        pred_inter_heatmap = inter_output[0, (cls==1)[0], :, :, :].unsqueeze(0)
        ### Stacked Hourglass 구조이기에, 중간 Output과 
        train_hm_loss = hm_criterion(pred_heatmap, resize_heatmap) + hm_criterion(pred_inter_heatmap, resize_half_heatmap)
        train_gd_loss = gd_criterion(output)

        ### heatmap 으로 부터 예측 중심점 구함.
        pred_center = hadamard_product(output.squeeze(0))

        ### 예측 중심점을 0~1 로 정규화 한게 norm_pred_center
        norm_pred_center = pred_center
        norm_pred_center[:,0] = norm_pred_center[:,0] / RESIZE_DEPTH
        norm_pred_center[:,1] = norm_pred_center[:,1] / RESIZE_HEIGHT
        norm_pred_center[:,2] = norm_pred_center[:,2] / RESIZE_WIDTH
        norm_pred_center = norm_pred_center.unsqueeze(0)

        ### 예측 중심점 간의 DR Loss 를 통해서 GT 가 없는 치아의 중심점도 주변과의 거리를 통해 위치하게끔 함.
        train_dr_loss = dr_criterion(norm_pred_center)

        norm_pred_center = norm_pred_center[0, (cls==1)[0], :].unsqueeze(0)
        train_pt_loss = pt_criterion(norm_pred_center, norm_resize_center)

        ### 예측 박스를 중심과 offset (대각선) 을 통해 얻음.
        norm_offset = offset[0, (cls==1)[0], :].unsqueeze(0)
        norm_offset = norm_offset.squeeze(0)
        norm_offset[:,0] = norm_offset[:,0] / RESIZE_DEPTH
        norm_offset[:,1] = norm_offset[:,1] / RESIZE_HEIGHT
        norm_offset[:,2] = norm_offset[:,2] / RESIZE_WIDTH
        norm_offset = norm_offset.unsqueeze(0)

        coor1 = norm_pred_center - norm_offset/2
        coor2 = norm_pred_center + norm_offset/2
        pred_box = torch.cat([coor1, coor2], dim=2)

        train_box_loss = box_criterion(pred_box, norm_resize_bbox)

        train_metal_loss = metal_score_criterion(score_output, gt_metal_score)
        train_metal_loss = METAL_WEIGHT * train_metal_loss
        
        ### 각 Loss 에 대한 가중치. 실험적으로 최적화 된 수치는 아님.
        train_hm_loss = 0.1 * train_hm_loss
        train_gd_loss = 1 * train_gd_loss
        train_box_loss = 0.1 * train_box_loss
        train_pt_loss = 0.1 * train_pt_loss
        train_dr_loss = 0.001 * train_dr_loss

        if epoch > METAL_EPOCH:
            train_loss = \
                train_gd_loss + \
                train_hm_loss + \
                train_box_loss + \
                train_pt_loss + \
                train_dr_loss + \
                train_metal_loss

            print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | DR loss : {} | Metal loss : {}'.format(
                idx+1, len(train_dataloader), train_loss.item(), train_hm_loss.item(), train_gd_loss.item(), train_box_loss.item(), train_pt_loss.item(), train_dr_loss.item(), train_metal_loss.item())
            )

        elif epoch > DR_EPOCH:
            train_loss = \
                train_gd_loss + \
                train_hm_loss + \
                train_box_loss + \
                train_pt_loss + \
                train_dr_loss

            print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | DR loss : {}'.format(
                idx+1, len(train_dataloader), train_loss.item(), train_hm_loss.item(), train_gd_loss.item(), train_box_loss.item(), train_pt_loss.item(), train_dr_loss.item())
            )

        else:
            train_loss = \
                train_gd_loss + \
                train_hm_loss + \
                train_box_loss + \
                train_pt_loss

            print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {}'.format(
                idx+1, len(train_dataloader), train_loss.item(), train_hm_loss.item(), train_gd_loss.item(), train_box_loss.item(), train_pt_loss.item())
            )

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses += train_loss.item()

    scheduler.step()

    valid_losses = 0.0
    model.eval()

    print()
    print("Validation start !!")
    print()
            
    with torch.no_grad():
        if epoch == (DR_EPOCH + 1):
            min_valid_loss = math.inf

        if epoch == (METAL_EPOCH + 1):
            min_valid_loss = math.inf

        for idx, data in enumerate(test_dataloader):
            person_id = data['person_id']
            flag = data['flag']
            cls = data['cls']

            resize_image = data['resize_image']
            resize_bbox = data['resize_bbox'].type(torch.float32)
            resize_center = data['resize_center'].type(torch.float32)
            resize_heatmap = data['resize_heatmap']
            resize_half_heatmap = data['resize_half_heatmap']
            gt_metal_score = data['metal_score'].squeeze(0).type(torch.float32)
            
            norm_resize_center = resize_center.squeeze(0)
            norm_resize_center[:,0] = norm_resize_center[:,0] / RESIZE_DEPTH
            norm_resize_center[:,1] = norm_resize_center[:,1] / RESIZE_HEIGHT
            norm_resize_center[:,2] = norm_resize_center[:,2] / RESIZE_WIDTH
            norm_resize_center = norm_resize_center.unsqueeze(0)

            norm_resize_bbox = resize_bbox.squeeze(0)
            norm_resize_bbox[:,0] = norm_resize_bbox[:,0] / RESIZE_DEPTH
            norm_resize_bbox[:,1] = norm_resize_bbox[:,1] / RESIZE_HEIGHT
            norm_resize_bbox[:,2] = norm_resize_bbox[:,2] / RESIZE_WIDTH
            norm_resize_bbox[:,3] = norm_resize_bbox[:,3] / RESIZE_DEPTH
            norm_resize_bbox[:,4] = norm_resize_bbox[:,4] / RESIZE_HEIGHT
            norm_resize_bbox[:,5] = norm_resize_bbox[:,5] / RESIZE_WIDTH
            norm_resize_bbox = norm_resize_bbox.unsqueeze(0)

            inter_output, output, offset, score_output = model(resize_image)

            pred_heatmap = output[0, (cls==1)[0], :, :, :].unsqueeze(0)
            pred_inter_heatmap = inter_output[0, (cls==1)[0], :, :, :].unsqueeze(0)
            val_hm_loss = hm_criterion(pred_heatmap, resize_heatmap) + hm_criterion(pred_inter_heatmap, resize_half_heatmap)
            val_gd_loss = gd_criterion(output)

            pred_center = hadamard_product(output.squeeze(0))

            norm_pred_center = pred_center
            norm_pred_center[:,0] = norm_pred_center[:,0] / RESIZE_DEPTH
            norm_pred_center[:,1] = norm_pred_center[:,1] / RESIZE_HEIGHT
            norm_pred_center[:,2] = norm_pred_center[:,2] / RESIZE_WIDTH
            norm_pred_center = norm_pred_center.unsqueeze(0)

            val_dr_loss = dr_criterion(norm_pred_center)

            norm_pred_center = norm_pred_center[0, (cls==1)[0], :].unsqueeze(0)
            val_pt_loss = pt_criterion(norm_pred_center, norm_resize_center)

            norm_offset = offset[0, (cls==1)[0], :].unsqueeze(0)
            norm_offset = norm_offset.squeeze(0)
            norm_offset[:,0] = norm_offset[:,0] / RESIZE_DEPTH
            norm_offset[:,1] = norm_offset[:,1] / RESIZE_HEIGHT
            norm_offset[:,2] = norm_offset[:,2] / RESIZE_WIDTH
            norm_offset = norm_offset.unsqueeze(0)

            coor1 = norm_pred_center - norm_offset/2
            coor2 = norm_pred_center + norm_offset/2
            pred_box = torch.cat([coor1, coor2], dim=2)

            val_box_loss = box_criterion(pred_box, norm_resize_bbox)

            val_metal_loss = metal_score_criterion(score_output, gt_metal_score)
            val_metal_loss = METAL_WEIGHT * val_metal_loss
            
            val_hm_loss = 0.1 * val_hm_loss
            val_gd_loss = 1 * val_gd_loss
            val_box_loss = 0.1 * val_box_loss
            val_pt_loss = 0.1 * val_pt_loss
            val_dr_loss = 0.001 * val_dr_loss

            if epoch > METAL_EPOCH:
                valid_loss = \
                    val_gd_loss + \
                    val_hm_loss + \
                    val_box_loss + \
                    val_pt_loss + \
                    val_dr_loss + \
                    val_metal_loss

                print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | DR loss : {} | Metal loss : {}'.format(
                    idx+1, len(test_dataloader), valid_loss.item(), val_hm_loss.item(), val_gd_loss.item(), val_box_loss.item(), val_pt_loss.item(), val_dr_loss.item(), val_metal_loss.item())
                )

            elif epoch > DR_EPOCH:
                valid_loss = \
                    val_gd_loss + \
                    val_hm_loss + \
                    val_box_loss + \
                    val_pt_loss + \
                    val_dr_loss

                print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | DR loss : {}'.format(
                    idx+1, len(test_dataloader), valid_loss.item(), val_hm_loss.item(), val_gd_loss.item(), val_box_loss.item(), val_pt_loss.item(), train_dr_loss.item())
                )

            else:
                valid_loss = \
                    val_gd_loss + \
                    val_hm_loss + \
                    val_box_loss + \
                    val_pt_loss

                print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {}'.format(
                    idx+1, len(test_dataloader), valid_loss.item(), val_hm_loss.item(), val_gd_loss.item(), val_box_loss.item(), val_pt_loss.item())
                )

            valid_losses += valid_loss


        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_loss / len(test_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(test_dataloader)}')
        
        valid_losses = valid_losses / len(test_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')
            
        print()

writer.flush()
writer.close()