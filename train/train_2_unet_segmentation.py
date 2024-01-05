import torch
import torch.nn as nn
import math
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from model.unet_segmentation_model import UNet3D
from dataloader.segmentation_dataset import get_tooth_train_val_test_Dataloaders
from transform.segmentation_transform import (train_transform_mask, val_transform_mask)

torch.manual_seed(123)

writer = SummaryWriter()
model = UNet3D(in_channels=1, num_classes=1)
model.cuda()
train_dataloader, val_dataloader, _ = get_tooth_train_val_test_Dataloaders(train_transforms= train_transform_mask, val_transforms=val_transform_mask, test_transforms= val_transform_mask)

criterion = nn.MSELoss()

optimizer = Adam(params=model.parameters(), lr=1e-4)
scheduler = MultiStepLR(optimizer, [40, 120, 360], gamma=0.1, last_epoch=-1)
min_valid_loss = math.inf


for epoch in range(TRAINING_EPOCH):
    train_loss = 0.0
    model.train()
    print()
    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        # if idx > 0:
        #     break
        image, mask = data['image'], data['mask']
        target = model(image)
        train_mse_loss = criterion(target, mask)
        train_loss_mask = train_mse_loss
        
        if idx % 100 == 0:
            print(' {} / {} => Mask loss : {}'.format(idx+1, len(train_dataloader), train_loss_mask.item()))

        train_loss_mask.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += train_loss_mask

    scheduler.step()
    valid_losses = 0.0
    model.eval()
    
    print()
    print("Validation start !!")
    print()

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            image, mask = data['image'], data['mask']
            target = model(image)
            val_mse_loss = criterion(target, mask)
            val_loss_mask = val_mse_loss

            if idx % 100 == 0:
                print(' {} / {} => Mask loss : {}'.format(idx+1, len(val_dataloader), val_loss_mask.item()))
            valid_losses += val_loss_mask

            if idx == len(val_dataloader) -1:
            # if idx == 0:
                temp_image = image
                temp_gt_mask = mask
                temp_target = target

        writer.add_scalar("Loss/Train_mask", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation_mask", valid_losses / len(val_dataloader), epoch)
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)}')
        valid_losses = valid_losses / len(val_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')
        print()

writer.flush()
writer.close()