dataset_type = 'DOTADataset'
data_root = 'data/split_ss_dota1_0/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOBBAnnotations',
        with_bbox=True,
        with_label=True,
        with_poly_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(
        type='RandomOBBRotate',
        rotate_after_flip=True,
        angles=(0, 0),
        vert_rate=0.5,
        vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type='OBBCollect',
        keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1024, 1024)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='DOTADataset',
        task='Task1',
        ann_file='data/split_ss_dota1_0/trainval/annfiles/',
        img_prefix='data/split_ss_dota1_0/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadOBBAnnotations',
                with_bbox=True,
                with_label=True,
                with_poly_as_mask=True),
            dict(type='LoadDOTASpecialInfo'),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='RandomOBBRotate',
                rotate_after_flip=True,
                angles=(0, 0),
                vert_rate=0.5,
                vert_cls=['roundabout', 'storage-tank']),
            dict(type='Pad', size_divisor=32),
            dict(type='DOTASpecialIgnore', ignore_size=2),
            dict(type='FliterEmpty'),
            dict(type='Mask2OBB', obb_type='obb'),
            dict(type='OBBDefaultFormatBundle'),
            dict(
                type='OBBCollect',
                keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
        ]),
    test=dict(
        type='DOTADataset',
        task='Task1',
        ann_file='data/split_ss_dota1_0/test/annfiles/',
        img_prefix='data/split_ss_dota1_0/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipRotateAug',
                img_scale=[(1024, 1024)],
                h_flip=False,
                v_flip=False,
                rotate=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='OBBRandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='RandomOBBRotate', rotate_after_flip=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='OBBCollect', keys=['img'])
                ])
        ]),
    val=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipRotateAug',
            img_scale=[(1024, 1024)],
            h_flip=False,
            v_flip=False,
            rotate=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='OBBRandomFlip'),
                dict(
                    type='Normalize',
                    mean=[102.9801, 115.9465, 122.7717],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='RandomOBBRotate', rotate_after_flip=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='OBBCollect', keys=['img'])
            ])
    ]))
evaluation = None
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
total_epochs = 12
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='FCOSOBB',
    pretrained='open-mmlab://detectron/resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='OBBFCOSHead',
        num_classes=15,
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
    max_per_img=2000)