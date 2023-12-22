from monai.transforms import (
    Compose,
    ToTensord,
    NormalizeIntensityd,
    AddChanneld,
)

train_detection_transform = Compose(
    [   
        AddChanneld(keys=["resize_image"]),
        NormalizeIntensityd(keys=["resize_image"], nonzero=True, channel_wise=True),
        ToTensord(keys=[
            'resize_image',
            'resize_bbox',
            'resize_center',
            'resize_heatmap',
            'resize_half_heatmap',
            'cls',
        ], device='cuda')
    ]
)

test_detection_transform = Compose(
    [   
        AddChanneld(keys=["resize_image"]),
        NormalizeIntensityd(keys=["resize_image"], nonzero=True, channel_wise=True),
        ToTensord(keys=[
            'resize_image',
            'resize_bbox',
            'resize_center',
            'resize_heatmap',
            'resize_half_heatmap',
            'cls',
        ], device='cuda')
    ]
)