_base_ = [
    '../../_base_/datasets/td.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../../_base_/default_runtime.py'
]
data_root = 'data/td/'
img_rescale_ratio = 0.25
img_scale=(3920*img_rescale_ratio, 2160*img_rescale_ratio)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = ('car','other_vehicle')
# model settings
model = dict(
    type='FCOSOBB',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='OBBFCOSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        scale_theta=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='PolyIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=200)
# optimizer
optimizer = dict(
    lr=0.005, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,keep_shape=False, #not keep shape will have more black edge
         angles=(-180, 180), vert_rate=0.5),
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
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        ann_file=data_root + 'split_set_train/annfiles2/*.pkl',
        img_prefix=data_root + 'split_set_train/images/',    
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        ann_file=data_root + 'split_set_test/annfiles2/*.pkl',
        img_prefix=data_root + 'split_set_test/images/',   
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        ann_file=data_root + 'split_set_test/annfiles2/*.pkl',
        img_prefix=data_root + 'split_set_test/images/',   
        pipeline=test_pipeline,
        classes=classes))

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
total_epochs = 24
work_dir = 'data/td/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_td_mixmorepatch_rotate'