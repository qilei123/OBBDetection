dataset_type = 'TDDataset'
data_root = 'data/td/'
img_rescale_ratio = 0.25
img_scale=(3920*img_rescale_ratio, 2160*img_rescale_ratio)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 0), vert_rate=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[img_scale],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]

# does evaluation while training
# uncomments it  when you need evaluate every epoch
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_AW_obb.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_AW_obb.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_AW_obb.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = dict(metric='mAP')

# disable evluation, only need train and test
# uncomments it when use trainval as train
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'Annotations/obb',
#         img_prefix=data_root + 'JPEGImages/',
#         pipeline=train_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'Annotations/obb',
#         img_prefix=data_root + 'JPEGImages/',
#         pipeline=test_pipeline))
# evaluation = None
