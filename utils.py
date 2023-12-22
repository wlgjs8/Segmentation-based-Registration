import torch
import numpy as np

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
        heatmaps : [1, N, 64, 128, 128]
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