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
    TRAINING_EPOCH, RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)
# from hourglass import HourGlass3D
from model.metal_concat_crop_no import HourGlass3D
from dataloader.detection_dataset import get_detection_dataloader
from transform.detection_transform import train_detection_transform, test_detection_transform
from loss.gaussian_disentangle import GDLoss
from loss.distance_regularization import DRLoss
from loss.losses import FocalLoss


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

hm_criterion = FocalLoss()
gd_criterion = GDLoss()
box_criterion = nn.MSELoss()
pt_criterion = nn.MSELoss()
dr_criterion = DRLoss()
# metal_score_criterion = nn.CrossEntropyLoss()

# weight = torch.tensor([1, 10]).cuda()
# metal_score_criterion = nn.BCELoss(weight=weight)
metal_score_criterion = nn.BCELoss()

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [100, 120, 360], gamma=0.1, last_epoch=-1)

min_valid_loss = math.inf
min_eval_err = math.inf
max_eval_iou = -1.

# DR_EPOCH = -1
DR_EPOCH = 100
# METAL_WEIGHT = 0.01
METAL_WEIGHT = 0.001
eps_smooth = 1e-6

for epoch in range(TRAINING_EPOCH):
    train_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    model.train()

    print()
    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        # if idx == 1:
        #     break
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

        inter_output, output, offset, pred_center, score_output = model(resize_image)
        # print('bef score_output : ', score_output.shape)

        # score_output = score_output[(cls==1)[0], :]
        print('score_output : ', score_output)
        print('gt_metal_score : ', gt_metal_score)
        # print('score_output : ', score_output.shape)
        # print('gt_metal_score : ', gt_metal_score.shape)
        print()
        train_metal_loss = metal_score_criterion(score_output, gt_metal_score)
        train_metal_loss = METAL_WEIGHT * train_metal_loss

        # score_output = score_output[(cls==1)[0], :]

        inter_pred_heatmap = inter_output[0, (cls==1)[0], :, :, :].unsqueeze(0)
        train_hm_loss1 = hm_criterion(inter_pred_heatmap, resize_half_heatmap)

        pred_heatmap = output[0, (cls==1)[0], :, :, :].unsqueeze(0)
        train_hm_loss2 = hm_criterion(pred_heatmap, resize_heatmap)
        train_gd_loss = gd_criterion(output)
        
        train_hm_loss = train_hm_loss1 + train_hm_loss2

        # pred_center = hadamard_product(output.squeeze(0))
        # norm_pred_center = pred_center[(cls==1)[0], :]
        norm_pred_center = pred_center
        norm_pred_center[:,0] = norm_pred_center[:,0] / RESIZE_DEPTH
        norm_pred_center[:,1] = norm_pred_center[:,1] / RESIZE_HEIGHT
        norm_pred_center[:,2] = norm_pred_center[:,2] / RESIZE_WIDTH
        norm_pred_center = norm_pred_center.unsqueeze(0)
        
        if epoch > DR_EPOCH:
            train_dr_loss = dr_criterion(norm_pred_center)
        
        norm_pred_center = norm_pred_center[0, (cls==1)[0], :].unsqueeze(0)
        train_pt_loss = pt_criterion(norm_pred_center, norm_resize_center)

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
        train_hm_loss = 10 * train_hm_loss
        train_gd_loss = 20 * train_gd_loss

        np_score_output = score_output.clone().detach().cpu().numpy()
        np_gt_metal_score = gt_metal_score.clone().detach().cpu().numpy()
        tp, fp, tn, fn, _ = eval_utils.cal_metal_tpfn_all(np_score_output, np_gt_metal_score)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        if epoch > DR_EPOCH:
            train_losses = \
                train_gd_loss + \
                train_hm_loss + \
                train_box_loss + \
                train_pt_loss + \
                train_dr_loss + \
                train_metal_loss

            print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | DR loss : {} | Metal loss : {}'.format(
                idx+1, len(train_dataloader), train_losses.item(), train_hm_loss.item(), train_gd_loss.item(), train_box_loss.item(), train_pt_loss.item(), train_dr_loss.item(), train_metal_loss.item())
            )

        else:
            train_losses = \
                train_gd_loss + \
                train_hm_loss + \
                train_box_loss + \
                train_pt_loss + \
                train_metal_loss

            print(' {} / {} => Total loss : {} | Heatmap loss : {} | GD loss : {} | Box loss : {} | PT loss : {} | Metal loss : {}'.format(
                idx+1, len(train_dataloader), train_losses.item(), train_hm_loss.item(), train_gd_loss.item(), train_box_loss.item(), train_pt_loss.item(), train_metal_loss.item())
            )

        # train_losses = train_metal_loss

        train_losses.backward()
        # _log_params(model, writer, idx)

        optimizer.step()
        optimizer.zero_grad()

        train_loss += train_losses.item()

    scheduler.step()

    valid_losses = 0.0
    model.eval()

    print()
    print("Validation start !!")
    print()
            
    with torch.no_grad():
        temp_image = None
        temp_inter_target = None
        temp_target = None
        
        temp_gt_cls = None
        temp_gt_center = None
        temp_gt_heatmap = None
        temp_gt_half_heatmap = None
        temp_person_id = None
        temp_flag = None

        # total_tp = 0
        # total_fp = 0
        # total_tn = 0
        # total_fn = 0

        point_error_heatmap_list = []
        box_iou_list = []

        if epoch == (DR_EPOCH + 1):
            min_valid_loss = math.inf

        for idx, data in enumerate(test_dataloader):
            person_id = data['person_id']
            flag = data['flag']
            cls = data['cls']

            resize_image = data['resize_image']
            resize_bbox = data['resize_bbox'].type(torch.float32)
            resize_center = data['resize_center']
            resize_heatmap = data['resize_heatmap']
            gt_metal_score = data['metal_score'].squeeze(0).type(torch.float32)

            norm_resize_center = resize_center.squeeze(0)
            norm_resize_center[:,0] = norm_resize_center[:,0] / RESIZE_DEPTH
            norm_resize_center[:,1] = norm_resize_center[:,1] / RESIZE_HEIGHT
            norm_resize_center[:,2] = norm_resize_center[:,2] / RESIZE_WIDTH
            norm_resize_center = norm_resize_center.unsqueeze(0)

            # norm_resize_bbox = resize_bbox.squeeze(0)
            norm_resize_bbox = resize_bbox.squeeze(0).detach().cpu().numpy()
            # norm_resize_bbox[:,0] = norm_resize_bbox[:,0] / RESIZE_DEPTH
            # norm_resize_bbox[:,1] = norm_resize_bbox[:,1] / RESIZE_HEIGHT
            # norm_resize_bbox[:,2] = norm_resize_bbox[:,2] / RESIZE_WIDTH
            # norm_resize_bbox[:,3] = norm_resize_bbox[:,3] / RESIZE_DEPTH
            # norm_resize_bbox[:,4] = norm_resize_bbox[:,4] / RESIZE_HEIGHT
            # norm_resize_bbox[:,5] = norm_resize_bbox[:,5] / RESIZE_WIDTH
            # norm_resize_bbox = norm_resize_bbox.unsqueeze(0)

            # inter_output, output, offset = model(resize_image)
            inter_output, output, offset, pred_center, score_output = model(resize_image)

            pred_heatmap = output[0, (cls==1)[0], :, :, :].unsqueeze(0)
            val_heatmap_center = get_maximum_point_tensor(pred_heatmap.squeeze(0))
            val_heatmap_center = val_heatmap_center.squeeze(0).detach().cpu().numpy()
            val_heatmap_point_err = eval_utils.point_error2(val_heatmap_center, norm_resize_center)
            point_error_heatmap_list.append(val_heatmap_point_err)

            pred_center = hadamard_product(output.squeeze(0))[(cls==1)[0], :]
            norm_offset = offset[0, (cls==1)[0], :].unsqueeze(0)
            coor1 = pred_center - norm_offset/2
            coor2 = pred_center + norm_offset/2
            pred_box = torch.cat([coor1, coor2], dim=2)

            pred_box = pred_box.squeeze(0).detach().cpu().numpy()

            val_box_iou = eval_utils.box_iou(pred_box, norm_resize_bbox)
            box_iou_list.append(val_box_iou)


            # score_output = score_output[(cls==1)[0], :]
            # score_output = score_output.detach().cpu().numpy()
            print('score_output : ', score_output)
            print('gt_metal_score : ', gt_metal_score)
            print()

            np_score_output = score_output.clone().detach().cpu().numpy()
            gt_metal_score = gt_metal_score.detach().cpu().numpy()
            tp, fp, tn, fn, _ = eval_utils.cal_metal_tpfn_all(np_score_output, gt_metal_score)

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

            print(' {} / {} => Heatmap Point Error : {} '.format(idx+1, len(test_dataloader), val_heatmap_point_err))
            print(' {} / {} => Box IoU : {} '.format(idx+1, len(test_dataloader), val_box_iou))

            val_hm_loss = hm_criterion(pred_heatmap, resize_heatmap)

            valid_loss = val_hm_loss
            valid_losses += valid_loss

            if idx == 2:
                temp_image = resize_image
                temp_inter_target = None
                temp_target = output
                
                temp_gt_cls = cls
                temp_gt_heatmap = resize_heatmap
                temp_gt_center = resize_center
                temp_gt_half_heatmap = None
                temp_person_id = person_id[0]
                temp_flag = flag[0]

                temp_reg_points = None
                temp_reg_heatmaps = None

        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_losses / len(test_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(test_dataloader)}')
        
        valid_losses = valid_losses / len(test_dataloader)
        cur_min_eval_err = sum(point_error_heatmap_list)/len(point_error_heatmap_list)
        cur_max_eval_iou = sum(box_iou_list)/len(box_iou_list)

        # print('Total tp fp tn fn : {} {} {} {}'.format(total_tp, total_fp, total_tn, total_fn))
        # precision = total_tp / (total_tp + total_fp + eps_smooth)
        # recall = total_tp / (total_tp + total_fn + eps_smooth)
        # print('Precision : {}, Recall : {}'.format(precision, recall))
        # cur_metal_acc = (total_tp + total_tn) / 219
        cur_metal_acc = total_tp / (total_tp + total_fp + total_fn + eps_smooth)

        if max_eval_iou < cur_max_eval_iou:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            # torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_acc{cur_metal_acc}.pth')
            print('min_eval_err : {} -> cur_min_eval_err : {}'.format(max_eval_iou, cur_max_eval_iou))
            min_eval_err = cur_min_eval_err
            max_eval_iou = cur_max_eval_iou

            save_points2(
                    temp_image, temp_inter_target, temp_target, 
                    temp_gt_cls, temp_gt_heatmap, temp_gt_center,
                    temp_reg_points, temp_reg_heatmaps, temp_person_id, temp_flag
                )
            
        # elif max_eval_iou < cur_max_eval_iou:
        #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
        #     min_valid_loss = valid_losses
        #     # Saving State Dict
        #     torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_iou_valLoss{min_valid_loss}.pth')
        #     print('max_eval_iou : {} -> cur_max_eval_iou : {}'.format(max_eval_iou, cur_max_eval_iou))
        #     max_eval_iou = cur_max_eval_iou

        #     save_points2(
        #             temp_image, temp_inter_target, temp_target, 
        #             temp_gt_cls, temp_gt_heatmap, temp_gt_center,
        #             temp_reg_points, temp_reg_heatmaps, temp_person_id, temp_flag
        #         )
            
        else:
            print('NoopNoop min_eval_err : {} -> cur_min_eval_err : {}'.format(min_eval_err, cur_min_eval_err))
            print('NoopNoop max_eval_iou : {} -> cur_max_eval_iou : {}'.format(max_eval_iou, cur_max_eval_iou))
            print('NoopNoop max_eval_iou : {}'.format(cur_metal_acc))
        print()

writer.flush()
writer.close()