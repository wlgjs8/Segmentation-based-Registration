from monai.transforms import (
    Compose,
    ToTensord,
    NormalizeIntensityd,
    AddChanneld,
)

train_transform_mask = Compose(
    [   
        AddChanneld(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=[
            'image',
            'mask',
            'whole_mask',
        ], device='cuda')
    ]
)

val_transform_mask = Compose(
    [   
        AddChanneld(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=[
            'image',
            'mask',
            'whole_mask',
        ], device='cuda')
    ]
)