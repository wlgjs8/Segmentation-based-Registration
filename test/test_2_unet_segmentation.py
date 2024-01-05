import torch
import torch.nn as nn
import math
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from unet3d_seg import UNet3D
from dataset_bbox import get_tooth_train_val_test_Dataloaders
from osstem_transforms import (train_transform_mask, val_transform_mask)

from save_wlgjs import save_mask
from losses import DiceScore


torch.manual_seed(123)

writer = SummaryWriter()
model = UNet3D(in_channels=1, num_classes=1)

MODEL_WEIGHT_PATH = './checkpoints_mask/epoch39_valLoss0.005633824970573187.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model.cuda()
train_dataloader, val_dataloader, _ = get_tooth_train_val_test_Dataloaders(train_transforms= train_transform_mask, val_transforms=val_transform_mask, test_transforms= val_transform_mask)

# criterion = nn.MSELoss()
criterion = DiceScore()

valid_scores = 0.0
model.eval()

print()
print("Validation start !!")
print()

with torch.no_grad():
    for idx, data in enumerate(val_dataloader):
        image, mask = data['image'], data['whole_mask']
        target = model(image)

        # target[target < 0.1] = 0

        thres = 0.2
        target[target < thres] = 0
        target[target >= thres] = 1

        val_dice_mask = criterion(target, mask)

        if idx < 10:
            temp_image = image
            temp_gt_mask = mask
            temp_target = target

            save_mask(temp_image, temp_target, temp_gt_mask, idx)

        temp_image = image
        temp_gt_mask = mask
        temp_target = target

        # save_mask(temp_image, temp_target, temp_gt_mask, idx)

        # print('image : ', image.shape)
        # print('target : ', target.shape)


        print(' {} / {} => Mask loss : {}'.format(idx+1, len(val_dataloader), val_dice_mask.item()))
        valid_scores += val_dice_mask

    print(f'Validation Loss: {valid_scores / len(val_dataloader)}')
    valid_scores = valid_scores / len(val_dataloader)

    print()