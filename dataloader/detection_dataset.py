import os
import nibabel as nib
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader

from utils import generate_gaussian_heatmap, resize_img
from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH, 
    UPPER_TOOTH_NUM, LOWER_TOOTH_NUM, OUTLIER
)
from config import DETECTION_TRAIN, DETECTION_TEST

class RealignedCT(Dataset):
    def __init__(self, transform, mode='train'):
        super(RealignedCT, self).__init__()

        self.dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/realigned")
        self.transform = transform
        self.mode = mode
        self.tooth_per_image = 16

        if self.mode == 'train':
            self.data = DETECTION_TRAIN
        else:
            self.data = DETECTION_TEST

        self.crop_data = []
        for single_data in self.data:
            self.crop_data.append((single_data, 'upper'))
            self.crop_data.append((single_data, 'lower'))

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, idx):
        person_id, flag = self.crop_data[idx]
        print('person_id : {}, flag : {}'.format(person_id, flag))

        '''
        1. 영상 경로 & 영상
        2. Box Anno 경로

        3. Resize
        4. Box Outlier 제거 (11, 12, 없을 수도 있음.)

        5. Box -> Center
        6. Center -> Gaussian
        7. 32 x 64 x 64 Gaussian
        8. Transform
        '''

        if flag == 'upper':
            img_name = 'crop_image_upper.nii.gz'
            box_anno = 'upper_bbox.json'
            TOOTH_NUM = UPPER_TOOTH_NUM
        elif flag == 'lower':
            img_name = 'crop_image.nii.gz'
            box_anno = 'lower_bbox.json'
            TOOTH_NUM = LOWER_TOOTH_NUM

        img_path = os.path.join(self.dir, person_id, img_name)
        img_object = nib.load(img_path)
        original_image = img_object.get_fdata()
        original_shape = original_image.shape

        box_anno_path = os.path.join(self.dir, person_id, box_anno)
        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)

        ban_list = []
        if person_id in OUTLIER.keys():
            ban_list = OUTLIER[person_id]

        cls = np.array([0 for _ in range(self.tooth_per_image)])
        resize_image = resize_img(original_image, size=(RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
        
        original_bbox_list = []
        original_center_list = []
        resize_bbox_list = []
        resize_center_list = []

        original_heatmap_list = []
        resize_heatmap_list = []
        resize_half_heatmap_list = []

        for ref_idx in range(self.tooth_per_image):
            tooth_idx = TOOTH_NUM[ref_idx]
            
            if tooth_idx in ban_list:
                # print('reject ban_list : ', tooth_idx)
                continue

            if tooth_idx not in bbox_anno.keys():
                # print('reject no box anno : ', tooth_idx)
                continue

            '''
            정상 치아 시작
            '''
            cls[ref_idx] = 1
            original_bbox = np.array(bbox_anno[tooth_idx])
            original_bbox_list.append(original_bbox)
            original_center = np.array([(original_bbox[0]+original_bbox[3])/2, (original_bbox[1]+original_bbox[4])/2, (original_bbox[2]+original_bbox[5])/2])
            original_center_list.append(original_center)
            
            resize_bbox = original_bbox / np.concatenate([original_shape, original_shape]) * np.array([RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH, RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH])
            resize_bbox_list.append(resize_bbox)
            resize_center = original_center / np.array(original_shape) * np.array([RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH])
            # resize_center = [int(resize_center[0]), int(resize_center[1]), int(resize_center[2])]
            resize_center_list.append(resize_center)

            radius = np.linalg.norm(resize_center - resize_bbox[0:3])
            resize_heatmap = generate_gaussian_heatmap((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH), resize_center, sigma=radius / 9)
            original_heatmap = resize_img(resize_heatmap, size=original_shape)
            resize_half_heatmap = resize_img(resize_heatmap, size=(RESIZE_DEPTH/2, RESIZE_HEIGHT/2, RESIZE_WIDTH/2))
            resize_heatmap_list.append(resize_heatmap)
            original_heatmap_list.append(original_heatmap)
            resize_half_heatmap_list.append(resize_half_heatmap)

        cls = np.array(cls)
        original_bbox_list = np.array(original_bbox_list)
        original_center_list = np.array(original_center_list)
        resize_bbox_list = np.array(resize_bbox_list)
        resize_center_list = np.array(resize_center_list)
        resize_heatmap_list = np.array(resize_heatmap_list)
        original_heatmap_list = np.array(original_heatmap_list)
        resize_half_heatmap_list = np.array(resize_half_heatmap_list)

        proccessed_out = {
            'person_id' : person_id,
            'flag' : flag,
            'resize_image' : resize_image,
            'resize_bbox' : resize_bbox_list,
            'resize_center' : resize_center_list,
            'resize_heatmap' : resize_heatmap_list,
            'resize_half_heatmap' : resize_half_heatmap_list,
            'cls' : cls,
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out


def get_detection_dataloader(train_transform, test_transform):
    train_dataset = RealignedCT(transform=train_transform, mode='train')
    test_dataset = RealignedCT(transform=test_transform, mode='val')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader