import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from skimage.transform import resize
from config import (
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)

import utils
from utils import resize_img, generate_gaussian_heatmap, voi_crop
from tqdm import tqdm

from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)

from save_wlgjs import save_mask

# TRAIN = ['6', '7', '8', '9', '12', '13', '15', '19', '23', '25', '26', '28', '29', '30', '31', '32',
#          '34', '35', '37', '38', '39', '41', '46', '47', '48', '50', '51', '52']
# TRAIN = ['51', '52']
# TRAIN = ['6', '7', '8', '9', '12', '13', '15', '19', '23', '25', '26', '28', '29', '30', '31', '32',
#          '33', '34', '35', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']
TRAIN = ['6', '7', '8', '9', '12', '13', '15', '19', '23', '26', '28', '29', '30', '31',
         '34', '35', '37', '38', '41', '44', '46', '47', '48', '49', '50', '51']
# 21, 33, 36, 44, 49, 50

# 10, 20, 25, 32, 43, 52, 39
# 20, 25, 32, 43, 52
VAL = ['1', '2', '3', '4', '5', '11', '14', '16', '17']
TEST = ['18', '22', '24', '27', '40', '42', '45']
# 20
# TEST = [ '1', '2', '3', '4', '5', '10', '11', '14', '16', '17', '6', '7', '8', '9', '12', '13', '15', '19', '21', '25', '26', '28', '29', '30', '31', '32', '33',
#          '34', '35', '36', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']

# TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18',
#              '21', '22', '23', '24', '25', '26', '27', '28',
#              '31', '32', '33', '34', '35', '36', '37', '38',
#              '41', '42', '43', '44', '45', '46', '47', '48']

# UPPER_TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18', 
#             '21', '22', '23', '24', '25', '26', '27', '28']
# LOWER_TOOTH_NUM = ['41', '42', '43', '44', '45', '46', '47', '48',
#                ]

UPPER_TOOTH_NUM = [
    '21', '22', '23', '24', '25', '26', '27', '28',
    '11', '12', '13', '14', '15', '16', '17', '18', 
]

LOWER_TOOTH_NUM = [
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]

MASK_NUM = [
    '21', '22', '23', '24', '25', '26', '27', '28',
    '11', '12', '13', '14', '15', '16', '17', '18',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]

OUTLIER = {'1' : ['34'],
           '2' : ['12', '13'],
           '3' : ['17', '34', '36', '42', '43', '44', '45', '47'],
           '4' : ['16', '23', '25', '35', '37', '42', '47'],
           '5' : ['35'],
           '10' : ['11'],
           '11' : ['11'],
           '14' : ['31'],
           '16' : ['42'],
           '17' : ['43'],
           '18' : ['13'],
           '22' : ['33'],
           '24' : ['13'],
           '27' : ['45'],
           '40' : ['16'],
           '42' : ['46'],
           '45' : ['25']}


SEGM_TRAIN = [
    '2', '3', '4', '5',
    '6', '7', 
                      '14', '15',
          '17', '18',
          '22', '23',
                      '29', '30',
    '31',       '33', '34',
          '37', '38',       

    '46', '47',             '50',
]
SEGM_TEST = [
    '1', '41', '8', '12', '13', '28', '35', '42',
    # '2'
]

'''
43번 환자의 Mask 데이터는 아예 존재하지 않음..
'''


class MedicalSegmentationDecathlon(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, transforms = None, mode = None) -> None:
        super(MedicalSegmentationDecathlon, self).__init__()
        # self.dir = "../datasets/osstem_clean"
        # self.dir = "F:/osstem_clean"
        # '''
        # C:\Users\JeeheonKim\source\ct\pose\realigned
        # '''
        # self.dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/realigned")
        self.dir = os.path.abspath("F:/osstem_clean2")
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        if self.mode == 'train':
            self.data = TRAIN + VAL
        else:
            self.data = TEST
            
        self.input = '1.nii.gz'
        # self.input = 'crop_image.nii.gz'
        self.box_anno = 'bbox.json'
        # self.box_anno = 'lower_bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        person_id = self.data[idx]
        print("< person", person_id, ">")
        
        file_list = sorted(os.listdir(os.path.join(self.dir, person_id, 'nii')))
        img_path = os.path.join(self.dir, person_id, 'nii', self.input)
        box_anno_path = os.path.join(self.dir, person_id, 'nii', self.box_anno)
        mask_list = set(file_list) - set([self.input, self.box_anno, self.whole_mask])
        if person_id in OUTLIER.keys():
            mask_list = mask_list - set(OUTLIER[person_id])
        mask_list = list(mask_list)
        
        img_object = nib.load(img_path)
        img_array = img_object.get_fdata()
        img_array = img_array.astype(int)
        img_array = torch.Tensor(img_array).permute(-1, 1, 0)
        d, h, w = img_array.shape
        img_array = torch.flip(img_array, [0, 1])
        # print("img :", img_array.shape)
        
        # image = nib.Nifti1Image(np.array(img_array), affine=np.eye(4))
        # nib.save(image, 'image_{}.nii.gz'.format(person_id))
        
        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)
            
        # Remove outlier teeth
        annos = list(bbox_anno.keys())
        if person_id in OUTLIER.keys():
            print("teeth")
            print(annos)
            annos = sorted(list(set(annos) - set(OUTLIER[person_id])))
            print(">>>", annos)
        
        cls, bbox, heatmaps, mask = [], [], [], []
        for tooth in tqdm(annos, desc='(Annotating...) ', ascii=' ='):
            cls.append(np.expand_dims((np.array(TOOTH_NUM)==tooth)*1, axis=0))
            
            box = bbox_anno[tooth]
            # bbox -> heatmap
            box = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH])
            bbox.append(np.expand_dims(box, axis=0))
            heatmap = generate_gaussian_heatmap((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH), box)
            heatmaps.append(np.expand_dims(heatmap, axis=0))
            
            mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', tooth+"_gt.nii.gz"))
            mask_array = mask_object.get_fdata()
            mask_array = resize_img(mask_array, (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
            mask.append(np.expand_dims(mask_array, axis=0))
        
        img_array = resize_img(np.array(img_array), (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
        np_cls = np.concatenate(cls, axis=0)
        np_bbox = np.concatenate(bbox, axis=0)
        np_heatmap = np.concatenate(heatmaps, axis=0)
        np_mask = np.concatenate(mask, axis=0)
        
        proccessed_out = {
            'person_id': person_id,
            'image': img_array,
            'cls': np_cls,
            'bbox': np_bbox,
            'heatmap': np_heatmap,
            'mask' : np_mask
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out


class RealignedCT(Dataset):
 
    def __init__(self, transforms = None, mode = None) -> None:
        super(RealignedCT, self).__init__()
        self.dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/realigned")
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        if self.mode == 'train':
            self.data = SEGM_TRAIN
        else:
            self.data = SEGM_TEST

        self.crop_data = []
        for single_data in self.data:
            self.crop_data.append((single_data, 'upper'))
            self.crop_data.append((single_data, 'lower'))

        self.box_anno = 'lower_bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, idx):
        person_id, flag = self.crop_data[idx]
        # print('Person : ', person_id, ' Flag : ', flag)

        if flag == 'upper':
            crop_image_name = 'crop_image_upper.nii.gz'
            self.box_anno = 'upper_bbox.json'
            TOOTH_NUM = UPPER_TOOTH_NUM
        elif flag == 'lower':
            crop_image_name = 'crop_image.nii.gz'
            self.box_anno = 'lower_bbox.json'
            TOOTH_NUM = LOWER_TOOTH_NUM

        file_list = os.listdir(os.path.join(self.dir, person_id))
        new_file_list = []
        for file_name in file_list:
            if file_name[3:8] == flag:
                new_file_list.append(file_name)
        file_list = sorted(new_file_list)

        # img_path = os.path.join(self.dir, person_id, 'nii', self.input)
        img_path = os.path.join(self.dir, person_id, crop_image_name)
        box_anno_path = os.path.join(self.dir, person_id, self.box_anno)

        # mask_list = set(file_list) - set([self.input, self.box_anno, self.whole_mask])
        mask_list = set(file_list)
        if person_id in OUTLIER.keys():
            mask_list = mask_list - set(OUTLIER[person_id])
        mask_list = list(mask_list)
        
        img_object = nib.load(img_path)
        img_array = img_object.get_fdata()
        # img_array = img_array.astype(int)
        # img_array = torch.Tensor(img_array).permute(-1, 1, 0)
        d, h, w = img_array.shape
        # img_array = torch.flip(img_array, [0, 1])
        # print("img :", img_array.shape)
        
        # image = nib.Nifti1Image(np.array(img_array), affine=np.eye(4))
        # nib.save(image, 'image_{}.nii.gz'.format(person_id))
        
        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)
            
        # Remove outlier teeth
        annos = list(bbox_anno.keys())
        if person_id in OUTLIER.keys():
            # print("teeth")
            # print(annos)
            annos = sorted(list(set(annos) - set(OUTLIER[person_id])))
            # print(">>>", annos)
        
        cls, bboxes, centers, heatmaps, mask = [], [], [], [], []
        # for tooth in tqdm(annos, desc='(Annotating...) ', ascii=' ='):
        #     print('annos : ', annos)
        # for tooth in annos:
        for i in range(16):
            tooth = TOOTH_NUM[i]
            each_tooth_file = os.path.join(self.dir, person_id, '{}_{}_crop.nii.gz'.format(tooth, flag))
            if not os.path.exists(each_tooth_file):
                continue

            # if person_id in OUTLIER.keys():
            if tooth not in annos:
                # print('>> person id : ', person_id, ', >> tooth id : ', tooth)
                continue

            cls.append(np.expand_dims((np.array(TOOTH_NUM)==tooth)*1, axis=0))
            
            box = bbox_anno[tooth]
            
            bbox = np.array(box) / np.concatenate([img_array.shape, img_array.shape])
            bboxes.append(np.expand_dims(bbox, axis=0))
            
            # bbox -> heatmap
            # box = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([64, 64, 64])
            center = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH])

            # box = box / np.array(img_array.shape) * np.array([64, 64, 64])

            centers.append(np.expand_dims(center, axis=0))
            # heatmap = generate_gaussian_heatmap((128, 128, 128), box)
            heatmap = generate_gaussian_heatmap((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH), center)
            heatmaps.append(np.expand_dims(heatmap, axis=0))
            
            # mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', tooth+"_gt.nii.gz"))
            # mask_object = nib.load(os.path.join(self.dir, person_id, tooth+"_{}_crop.nii.gz".format(flag)))
            # mask_array = mask_object.get_fdata()
            # mask_array = resize_img(mask_array, (128, 128, 128))
            # mask.append(np.expand_dims(mask_array, axis=0))
        
        # img_array = resize_img(np.array(img_array), (128, 128, 128))
        # img_array = resize_img(img_array, (128, 128, 128))
        img_array = resize_img(img_array, (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
        np_cls = np.concatenate(cls, axis=0)
        np_bbox = np.concatenate(bboxes, axis=0)
        np_center = np.concatenate(centers, axis=0)
        np_heatmap = np.concatenate(heatmaps, axis=0)
        # np_mask = np.concatenate(mask, axis=0)
        
        '''
        영상 : [1, 25, 32]

        상악 : [1, 25, 32 or 16] : 
        하악 : [1, 25, 32 or 16] : [0, 0, 0, 0, ] + [1, 0, 0 ,0 ,0 ]


        '''

        # person_id, flag = self.crop_data[idx]

        proccessed_out = {
            'person_id': person_id,
            'image': img_array,
            'cls': np_cls,
            'bbox': np_bbox,
            'center': np_center,
            'heatmap': np_heatmap,
            # 'mask' : np_mask
            'flag' : flag,
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out


def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    train_dataset = RealignedCT(transforms=train_transforms, mode='train')
    val_dataset = RealignedCT(transforms=val_transforms, mode='val')

    train_dataloader = DataLoader(dataset= train_dataset, batch_size= TRAIN_BATCH_SIZE, shuffle= True)
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    # test_dataloader = DataLoader(dataset= test_set, batch_size= TEST_BATCH_SIZE, shuffle= False)
    test_dataloader = None
    
    return train_dataloader, val_dataloader, test_dataloader

def get_val_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    val_dataset = RealignedCT(transforms=val_transforms, mode='val')

    #Spliting dataset and building their respective DataLoaders
    # val_set = copy.deepcopy(dataset)
    # val_set.set_mode('val')
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    
    return val_dataloader


class RealignedTooth(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, transforms = None, mode = None) -> None:
        super(RealignedTooth, self).__init__()
        # self.dir = os.path.abspath("../datasets/realigned_2/realigned")
        self.dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/realigned")
        self.box_anno_dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/score_crop_dist")

        self.meta = dict()
        self.transform = transforms
        self.mode = mode
        if self.mode == 'train':
            self.data = SEGM_TRAIN
        else:
            self.data = SEGM_TEST
        self.crop_data = []
        for single_data in self.data:
            self.crop_data.append((single_data, 'upper'))
            self.crop_data.append((single_data, 'lower'))
        self.box_anno = 'lower_bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'
        self.tooth_lists = []

        print('SEGM_TRAIN : ', len(SEGM_TRAIN))
        print('SEGM_TEST : ', len(SEGM_TEST))

        for person_id, flag in self.crop_data:
            box_anno_path = os.path.join(self.box_anno_dir, person_id, 'box_pred_{}.json'.format(flag))
            with open(box_anno_path, 'r') as file:
                bbox_anno = json.load(file)
            # tooth_list = os.listdir(os.path.join(self.dir, person_id))
            # self.box_anno1 = 'box_pred_upper.json'
            # self.box_anno2 = 'box_pred_lower.json'

            for tidx in MASK_NUM:
                # if str(tidx) in bbox_anno.keys():
                #     # print('person_id : ', person_id)
                #     # print('tidx : ', tidx)
                #     self.tooth_lists.append((os.path.join(person_id, str(tidx)), flag))
                if tidx in bbox_anno.keys():
                    # print('person_id : ', person_id)
                    # print('tidx : ', tidx)
                    self.tooth_lists.append((os.path.join(person_id, tidx), flag))

        self.tooth_lists = list(set(self.tooth_lists))
        self.tooth_lists = sorted(self.tooth_lists)

    def __len__(self):
        return len(self.tooth_lists)

    def __getitem__(self, idx):
        tooth_dir, flag = self.tooth_lists[idx]
        # print('tooth_dir : ', tooth_dir)
        # person_id = tooth_dir.split('/')[0]
        person_id = tooth_dir.split('\\')[0]
        # print('person_id : ', person_id)
        tooth_id = (tooth_dir.split('\\')[1])

        # print('person_id : {}, tooth_id : {}'.format(person_id, tooth_id))

        # print('>> person_id : {}, tooth_id : {}'.format(person_id, tooth_id))
        # if 'upper' in tooth_dir:
        if flag == 'upper':
            crop_image_name = 'crop_image_upper.nii.gz'
            self.box_anno = 'box_pred_upper.json'

        elif flag == 'lower':
            crop_image_name = 'crop_image.nii.gz'
            self.box_anno = 'box_pred_lower.json'

        # img_path = os.path.join(self.dir, tooth_dir)
        # img_path = os.path.join(self.dir, str(person_id), tooth_id+flag)
        img_path = os.path.join(self.dir, str(person_id), crop_image_name)
        # box_anno_path = os.path.join(self.dir, person_id, self.box_anno)
        box_anno_path = os.path.join(self.box_anno_dir, person_id, self.box_anno)

        img_object = nib.load(img_path)
        img_array = img_object.get_fdata()
        img_array = resize_img(img_array, (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))

        # mask_object = nib.load(os.path.join(self.dir, person_id, '{}_{}_dist.nii.gz'.format(tooth_id, flag)))
        mask_object = nib.load(os.path.join(self.box_anno_dir, person_id, '{}_{}_dist.nii.gz'.format(tooth_id, flag)))
        mask_array = mask_object.get_fdata()
        mask_array = resize_img(mask_array, (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))



        mask_whole_object = nib.load(os.path.join(self.dir, person_id, '{}_{}_crop.nii.gz'.format(tooth_id, flag)))
        mask_whole_array = mask_whole_object.get_fdata()
        mask_whole_array = resize_img(mask_whole_array, (RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))

        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)

        # if str(tooth_id) not in bbox_anno.keys():
        #     continue
        
        bbox_anno = bbox_anno[tooth_id]
        bbox_anno = np.array(bbox_anno).astype(int)
        # print('bbox_anno : ', bbox_anno)

        # RE_MARGIN = 7
        # bbox_anno[0] = bbox_anno[0] + RE_MARGIN
        # bbox_anno[1] = bbox_anno[1] + RE_MARGIN
        # bbox_anno[2] = bbox_anno[2] + RE_MARGIN
        # bbox_anno[3] = bbox_anno[3] - RE_MARGIN
        # bbox_anno[4] = bbox_anno[4] - RE_MARGIN
        # bbox_anno[5] = bbox_anno[5] - RE_MARGIN

        # print('bef bbox_anno : ', bbox_anno)

        MARGIN = 3
        bbox_anno[0] = bbox_anno[0] - MARGIN
        bbox_anno[1] = bbox_anno[1] - MARGIN
        bbox_anno[2] = bbox_anno[2] - MARGIN
        bbox_anno[3] = bbox_anno[3] + MARGIN
        bbox_anno[4] = bbox_anno[4] + MARGIN
        bbox_anno[5] = bbox_anno[5] + MARGIN

        if bbox_anno[0] < 0:
            bbox_anno[0] = 0
        if bbox_anno[1] < 0:
            bbox_anno[1] = 0
        if bbox_anno[2] < 0:
            bbox_anno[2] = 0
        if bbox_anno[3] > 63:
            bbox_anno[3] = 63
        if bbox_anno[4] > 127:
            bbox_anno[4] = 127
        if bbox_anno[5] > 127:
            bbox_anno[5] = 127

        # print('aft bbox_anno : ', bbox_anno)
        # print()


        cropped_img = voi_crop(torch.Tensor(img_array).unsqueeze(0).unsqueeze(0),
                               (bbox_anno[0], bbox_anno[3]),
                               (bbox_anno[1], bbox_anno[4]),
                               (bbox_anno[2], bbox_anno[5]))

        cropped_mask = voi_crop(torch.Tensor(mask_array).unsqueeze(0).unsqueeze(0),
                                (bbox_anno[0], bbox_anno[3]),
                                (bbox_anno[1], bbox_anno[4]),
                                (bbox_anno[2], bbox_anno[5]))
        
        cropped_whole_mask = voi_crop(torch.Tensor(mask_whole_array).unsqueeze(0).unsqueeze(0),
                                (bbox_anno[0], bbox_anno[3]),
                                (bbox_anno[1], bbox_anno[4]),
                                (bbox_anno[2], bbox_anno[5]))

        # save_mask(
        #     torch.Tensor(img_array).unsqueeze(0).unsqueeze(0), 
        #     torch.Tensor(mask_array).unsqueeze(0).unsqueeze(0), 
        #     cropped_mask, 
        #     100
        # )

        cropped_img = resize_img(cropped_img[0][0], (128, 64, 64))
        cropped_mask = resize_img(cropped_mask[0][0], (128, 64, 64), 'nearest')
        cropped_whole_mask = resize_img(cropped_whole_mask[0][0], (128, 64, 64), 'nearest')

        # save_mask(
        #     cropped_img, 
        #     torch.Tensor(cropped_mask).unsqueeze(0).unsqueeze(0), 
        #     cropped_mask, 
        #     999
        # )

        proccessed_out = {
            'image': cropped_img.numpy(),
            'mask' : cropped_mask.numpy(),
            'whole_mask' : cropped_whole_mask.numpy()
        }
        proccessed_out = self.transform(proccessed_out)
        return proccessed_out

def get_tooth_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    Note: all the configs to generate dataloaders in included in "config.py"
    """
    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    train_dataset = RealignedTooth(transforms=train_transforms, mode='train')
    val_dataset = RealignedTooth(transforms=val_transforms, mode='val')
    train_dataloader = DataLoader(dataset= train_dataset, batch_size= TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    # test_dataloader = DataLoader(dataset= test_set, batch_size= TEST_BATCH_SIZE, shuffle= False)
    test_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader