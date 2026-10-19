# MS_tiny_rebar.py -- Mobile-Seed (AFFormer_for_MS_tiny) adapted for binary rebar data.
#
# Official model structure preserved unchanged:
#   backbone : AFFormer_for_MS_tiny
#   heads    : BoundaryHead + RefineHead (AFD inside)
#   pretrained: ./ckpt/AFFormer_tiny_ImageNet1k.pth
#
# Task-interface changes only:
#   - dataset     : RebarDataset_boundary (binary rebar seg + edge)
#   - pipeline    : LoadRebarAnnotations (no np.unpackbits)
#   - img_scale   : (640, 480) matching other baselines_clean experiments
#   - crop_size   : (480, 640) full 640x480 input, no cropping loss
#   - Resize      : keep_ratio=False, exact 640x480 (no random scale jitter)
#   - Pad         : seg_pad_val=0, sebound_pad_val=0  (binary tasks, no ignore index)
#   - RefineHead  : num_classes=1, loss_decode=ML_BCELoss(use_sigmoid=True)
#   - runner      : max_iters=5500 (50 epochs * 110 iters/epoch; 440 train / batch=4)
#   - checkpoint  : saved every 550 iters (5 epochs); eval every 550 iters
#   - batch       : samples_per_gpu=4 (single GPU)

_base_ = [
    '../_base_/models/Mobile_Seed.py',
    '../_base_/default_runtime.py',
]

# ---------------------------------------------------------------------------
# Model -- official architecture, binary output
# ---------------------------------------------------------------------------
model = dict(
    type='EncoderDecoderRefine',
    down_ratio=1,
    pretrained='./ckpt/AFFormer_tiny_ImageNet1k.pth',
    backbone=dict(
        type='AFFormer_for_MS_tiny',
        drop_path_rate=0.1,
    ),
    decode_head=[
        dict(
            type='BoundaryHead',
            bound_channels=[16, 16, 32, 32],
            bound_ratio=2,
            in_channels=[16, 64, 216, 216],
            in_index=[1, 3, 5, 6],
            channels=16 + 16 + 32 + 32,
            num_classes=1,
            loss_decode=dict(
                type='ML_BCELoss',
                use_sigmoid=True,
                loss_weight=1.0,
                loss_name='loss_be',
            ),
        ),
        dict(
            type='RefineHead',
            fuse_channel=96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=1,
            loss_decode=dict(
                type='ML_BCELoss',
                use_sigmoid=True,
                loss_weight=1.0,
                loss_name='loss_ce',
            ),
        ),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ---------------------------------------------------------------------------
# Dataset -- RebarDataset_boundary + LoadRebarAnnotations
# ---------------------------------------------------------------------------
dataset_type = 'RebarDataset_boundary'
data_root = 'data_ms'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# img_scale (W, H) for Resize; images are 640x480 in our dataset.
# Resize to exact input size; no random scale jitter (small dataset, uniform size).
img_scale = (640, 480)
crop_size = (480, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRebarAnnotations'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # seg_pad_val=0: background; sebound_pad_val=0: no edge on padding border
    dict(type='RebarPad', size=crop_size, pad_val=0, seg_pad_val=0, sebound_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_semantic_sebound']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='segmentation',
        sebound_dir='edge',
        img_suffix='.png',
        seg_map_suffix='.png',
        sebound_map_suffix='.png',
        split='splits/train.txt',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='segmentation',
        sebound_dir='edge',
        img_suffix='.png',
        seg_map_suffix='.png',
        sebound_map_suffix='.png',
        split='splits/val.txt',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='segmentation',
        sebound_dir='edge',
        img_suffix='.png',
        seg_map_suffix='.png',
        sebound_map_suffix='.png',
        split='splits/val.txt',
        pipeline=test_pipeline,
    ),
)

# ---------------------------------------------------------------------------
# Optimiser and schedule
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# 50 epochs: 440 train samples / batch=4 = 110 iters/epoch * 50 = 5500 iters
runner = dict(type='IterBasedRunner', max_iters=5500)
checkpoint_config = dict(by_epoch=False, interval=550)
evaluation = dict(interval=550, metric='mIoU', pre_eval=True)

find_unused_parameters = True
