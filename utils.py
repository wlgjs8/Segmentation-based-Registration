import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)

def generate_gaussian_heatmap(size, coord, sigma=2):
    '''
    input : 
        size : target heatmap size (e.g. (64, 128, 128))
        coord : center corrdinates (float)
    output : 
        heatmap : 3D Numpy Tensor
    '''
    d = np.arange(size[0])
    w = np.arange(size[1])
    h = np.arange(size[2])
    
    dx, wx, hx = np.meshgrid(d, w, h)
    heatmap = np.exp(-((dx-coord[0])**2 + (wx-coord[1])**2 + (hx-coord[2])**2) / (2*sigma**2))

    heatmap = np.transpose(heatmap, (1, 0, 2))
    # heatmap = torch.tensor(heatmap)

    return heatmap


def resize_img(img, size, mode='bilinear'):
    '''
    input : 
        img : 3D Numpy Array
        size : target resize shape
    output : 
        img : 3D Numpy Tensor
    '''
    depth, height, width = size
    depth = int(depth)
    height = int(height)
    width = int(width)
    d = torch.linspace(-1,1,depth)
    h = torch.linspace(-1,1,height)
    w = torch.linspace(-1,1,width)
    
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3)
    grid = grid.unsqueeze(0)

    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = torch.nn.functional.grid_sample(img, grid, mode=mode, align_corners=True)
    img = img.squeeze(0).squeeze(0)
    img = img.numpy()

    return img


def hadamard_product(heatmaps):
    '''
    input : 
        heatmaps : [N, 64, 128, 128]
    output : 
        results : [N, 3] weighted sum for center points (float)
    '''
    results = []
    heatmaps = heatmaps.unsqueeze(0).unsqueeze(-1)

    for i in range(heatmaps.shape[1]):
        single_heatmap = heatmaps[0, i]
        size = single_heatmap.shape

        d = torch.linspace(0, size[0]-1, size[0])
        h = torch.linspace(0, size[1]-1, size[1])
        w = torch.linspace(0, size[2]-1, size[2])

        meshz, meshy, meshx = torch.meshgrid((d,h,w))
        grid = torch.stack((meshz, meshy, meshx), 3).cuda()

        sum = torch.sum(single_heatmap)
        repeat_single_heatmap = single_heatmap.repeat(1, 1, 1, 3)
        res = repeat_single_heatmap * grid

        d_sum = torch.sum(res[:,:,:,0])
        h_sum = torch.sum(res[:,:,:,1])
        w_sum = torch.sum(res[:,:,:,2])

        pred_keypoints = torch.stack([(d_sum/sum), (h_sum/sum), (w_sum/sum)], dim=0)
        results.append(pred_keypoints)

    results = torch.stack(results, dim=0)
    return results


def feature_crop(features, points, box_size=[64, 32, 32], final_size=[64, 32, 32], margin=False):
    '''
    input : 
        features : [1, N, 64, 128, 128] Input 영상과 Stack1의 output 을 Concat한 feature map 
        points : [N, 3] 예측 중심점
        box_size : [64, 32, 32] crop 할 사이즈
        final_size : [64, 32, 32] crop 후에 Classifier 에 입력 사이즈
        margin : crop 박스에 Margin을 2 Voxel 만큼 줄지 말지 여부 (bool)
    output : 
        results : [N, 3] weighted sum for center points (float)
    '''
    margin_value = 2
    box_size = np.array(box_size)
    centers = points.detach().cpu().numpy()

    if margin == False:
        ds = centers[:, 0] - box_size[0]//2
        de = centers[:, 0] + box_size[0]//2
        hs = centers[:, 1] - box_size[1]//2
        he = centers[:, 1] + box_size[1]//2
        ws = centers[:, 2] - box_size[2]//2
        we = centers[:, 2] + box_size[2]//2
    else:
        ds = centers[:, 0] - box_size[0]//2 - margin_value
        de = centers[:, 0] + box_size[0]//2 + margin_value
        hs = centers[:, 1] - box_size[1]//2 - margin_value
        he = centers[:, 1] + box_size[1]//2 + margin_value
        ws = centers[:, 2] - box_size[2]//2 - margin_value
        we = centers[:, 2] + box_size[2]//2 + margin_value
        
    ds[ds<0] = 0
    de[de>=RESIZE_DEPTH] = RESIZE_DEPTH-1
    hs[hs<0] = 0
    he[he>=RESIZE_HEIGHT] = RESIZE_HEIGHT-1
    ws[ws<0] = 0
    we[we>=RESIZE_WIDTH] = RESIZE_WIDTH-1

    ds = ds.astype(int)
    de = de.astype(int)
    hs = hs.astype(int)
    he = he.astype(int)
    ws = ws.astype(int)
    we = we.astype(int)

    crop_features = []
    for i in range(len(centers)):
        crop_feature = features[:, :, ds[i]:de[i], hs[i]:he[i], ws[i]:we[i]]
        upsampled_crop_feature = nn.Upsample(size=final_size)(crop_feature)
        upsampled_crop_feature = upsampled_crop_feature.squeeze(0)
        crop_features.append(upsampled_crop_feature)

    crop_features = torch.stack(crop_features, dim=0)
    return crop_features


def draw_center(center_coords, shape=(RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH)):
    center_volumes = []
    for cen_idx in range(center_coords.shape[0]):
        center_coord = center_coords[cen_idx].astype(int)

        center_volume = np.zeros(shape)
        center_volume[center_coord[0], center_coord[1], center_coord[2]] = 1
        center_volumes.append(center_volume)

    return np.array(center_volumes)


def draw_box(box_coords, shape=(RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH)):
    box_volumes = []
    for box_idx in range(box_coords.shape[0]):
        box_coord = box_coords[box_idx].astype(int)

        box_volume = np.zeros(shape)
        box_volume[box_coord[0]:box_coord[3], box_coord[1], box_coord[2]] = 1
        box_volume[box_coord[0]:box_coord[3], box_coord[1], box_coord[5]] = 1
        box_volume[box_coord[0]:box_coord[3], box_coord[4], box_coord[2]] = 1
        box_volume[box_coord[0]:box_coord[3], box_coord[4], box_coord[5]] = 1
        
        box_volume[box_coord[0], box_coord[1]:box_coord[4], box_coord[2]] = 1
        box_volume[box_coord[0], box_coord[1]:box_coord[4], box_coord[5]] = 1
        box_volume[box_coord[3], box_coord[1]:box_coord[4], box_coord[2]] = 1
        box_volume[box_coord[3], box_coord[1]:box_coord[4], box_coord[5]] = 1
        
        box_volume[box_coord[0], box_coord[1], box_coord[2]:box_coord[5]] = 1
        box_volume[box_coord[0], box_coord[4], box_coord[2]:box_coord[5]] = 1
        box_volume[box_coord[3], box_coord[1], box_coord[2]:box_coord[5]] = 1
        box_volume[box_coord[3], box_coord[4], box_coord[2]:box_coord[5]] = 1

        box_volumes.append(box_volume)

    return np.array(box_volumes)


def draw_metal_boxes(box_coords, pred_metal_list, shape=(RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH)):
    box_volumes = []
    for box_idx in range(box_coords.shape[0]):
        box_coord = box_coords[box_idx].astype(int)

        if pred_metal_list[box_idx] == 1:
            color = 2
        else:
            color = 1

        box_volume = np.zeros(shape)
        box_volume[box_coord[0]:box_coord[3], box_coord[1], box_coord[2]] = color
        box_volume[box_coord[0]:box_coord[3], box_coord[1], box_coord[5]] = color
        box_volume[box_coord[0]:box_coord[3], box_coord[4], box_coord[2]] = color
        box_volume[box_coord[0]:box_coord[3], box_coord[4], box_coord[5]] = color
        
        box_volume[box_coord[0], box_coord[1]:box_coord[4], box_coord[2]] = color
        box_volume[box_coord[0], box_coord[1]:box_coord[4], box_coord[5]] = color
        box_volume[box_coord[3], box_coord[1]:box_coord[4], box_coord[2]] = color
        box_volume[box_coord[3], box_coord[1]:box_coord[4], box_coord[5]] = color
        
        box_volume[box_coord[0], box_coord[1], box_coord[2]:box_coord[5]] = color
        box_volume[box_coord[0], box_coord[4], box_coord[2]:box_coord[5]] = color
        box_volume[box_coord[3], box_coord[1], box_coord[2]:box_coord[5]] = color
        box_volume[box_coord[3], box_coord[4], box_coord[2]:box_coord[5]] = color
        
        box_volumes.append(box_volume)

    return np.array(box_volumes)


def save_file(array, save_path, save_dir, idx=None):
    nii_array = nib.Nifti1Image(array, affine=np.eye(4))
    if idx == None:
        nib.save(nii_array, save_dir + '{}.nii.gz'.format(save_path))
    else:
        nib.save(nii_array, save_dir + '{}_{}.nii.gz'.format(save_path, idx))


def save_detection_result(gt_image, 
                          gt_heatmap, gt_center, gt_bbox, gt_cls,
                          pred_heatmap, pred_center, pred_bbox,
                          save_dir='./results/detection/'):
    
    '''
    input : 
        gt_image : [64, 128, 128]
        gt_heatmap : [N, 64, 128, 128]
        gt_center : [N, 3]
        gt_bbox : [N, 6]
        gt_cls : [16]

        pred_heatmap : [16, 64, 128, 128]
        pred_center : [16, 3]
        pred_bbox : [16, 3]

    output : 
        results : [N, 3] weighted sum for center points (float)
    '''

    gt_bbox = draw_box(gt_bbox)
    gt_center = draw_center(gt_center)
    filtered_pred_heatmap = pred_heatmap[gt_cls==1]
    filtered_pred_center = draw_center(pred_center[gt_cls==1])
    filtered_pred_bbox = draw_box(pred_bbox[gt_cls==1])

    save_file(gt_image, 'image', save_dir)

    for idx in range(gt_heatmap.shape[0]):
        save_file(gt_heatmap[idx], 'gt_heatmap', save_dir, idx)
        save_file(gt_center[idx], 'gt_center', save_dir, idx)
        save_file(gt_bbox[idx], 'gt_bbox', save_dir, idx)
        save_file(filtered_pred_heatmap[idx], 'pred_heatmap', save_dir, idx)
        save_file(filtered_pred_center[idx], 'pred_center', save_dir, idx)
        save_file(filtered_pred_bbox[idx], 'pred_bbox', save_dir, idx)

def save_detection_and_metal_classification_result(gt_image, 
                          gt_heatmap, gt_center, gt_bbox, gt_cls,
                          pred_heatmap, pred_center, pred_bbox, pred_metal_list,
                          save_dir='./results/detection_and_metal_classification/'):
    
    '''
    input : 
        gt_image : [64, 128, 128]
        gt_heatmap : [N, 64, 128, 128]
        gt_center : [N, 3]
        gt_bbox : [N, 6]
        gt_cls : [16]

        pred_heatmap : [16, 64, 128, 128]
        pred_center : [16, 3]
        pred_bbox : [16, 3]

    output : 
        results : [N, 3] weighted sum for center points (float)
    '''

    gt_bbox = draw_metal_boxes(gt_bbox, pred_metal_list)
    gt_center = draw_center(gt_center)
    filtered_pred_heatmap = pred_heatmap[gt_cls==1]
    filtered_pred_center = draw_center(pred_center[gt_cls==1])
    filtered_pred_bbox = draw_metal_boxes(pred_bbox[gt_cls==1], pred_metal_list[gt_cls==1])

    save_file(gt_image, 'image', save_dir)

    for idx in range(gt_heatmap.shape[0]):
        save_file(gt_heatmap[idx], 'gt_heatmap', save_dir, idx)
        save_file(gt_center[idx], 'gt_center', save_dir, idx)
        save_file(gt_bbox[idx], 'gt_bbox', save_dir, idx)
        save_file(filtered_pred_heatmap[idx], 'pred_heatmap', save_dir, idx)
        save_file(filtered_pred_center[idx], 'pred_center', save_dir, idx)
        save_file(filtered_pred_bbox[idx], 'pred_bbox', save_dir, idx)