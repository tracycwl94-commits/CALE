# CALE inference configuration; frozen checkpoint, no detector retraining.
custom_imports = {'imports': ['cale'], 'allow_failed_imports': False}

default_scope = 'mmdet'

model = {'backbone': {'depth': 50,
              'frozen_stages': 1,
              'init_cfg': None,
              'norm_cfg': {'requires_grad': True, 'type': 'BN'},
              'norm_eval': True,
              'num_stages': 4,
              'out_indices': (0, 1, 2, 3),
              'style': 'pytorch',
              'type': 'ResNet'},
 'bbox_head': {'anchor_generator': {'octave_base_scale': 8,
                                    'ratios': [1.0],
                                    'scales_per_octave': 1,
                                    'strides': [8, 16, 32, 64, 128],
                                    'type': 'AnchorGenerator'},
               'bbox_coder': {'target_means': [0.0, 0.0, 0.0, 0.0],
                              'target_stds': [0.1, 0.1, 0.2, 0.2],
                              'type': 'DeltaXYWHBBoxCoder'},
               'feat_channels': 256,
               'in_channels': 256,
               'loss_bbox': {'loss_weight': 2.0, 'type': 'GIoULoss'},
               'loss_centerness': {'loss_weight': 1.0,
                                   'type': 'CrossEntropyLoss',
                                   'use_sigmoid': True},
               'loss_cls': {'alpha': 0.25,
                            'gamma': 2.0,
                            'loss_weight': 1.0,
                            'type': 'FocalLoss',
                            'use_sigmoid': True},
               'num_classes': 1203,
               'stacked_convs': 4,
               'type': 'ATSSCALEHead',
               'frequency_file': 'stat_files/lvis_frequency.csv',
               'prior_file': 'stat_files/lvis_lacunarity.csv',
               'eta': 0.8,
               'gamma': 2.0},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_size_divisor': 32,
                       'std': [58.395, 57.12, 57.375],
                       'type': 'DetDataPreprocessor'},
 'neck': {'add_extra_convs': 'on_output',
          'in_channels': [256, 512, 1024, 2048],
          'num_outs': 5,
          'out_channels': 256,
          'start_level': 1,
          'type': 'FPN'},
 'test_cfg': {'max_per_img': 100,
              'min_bbox_size': 0,
              'nms': {'iou_threshold': 0.6, 'type': 'nms'},
              'nms_pre': 1000,
              'score_thr': 0.0001},
 'train_cfg': {'allowed_border': -1,
               'assigner': {'topk': 9, 'type': 'ATSSAssigner'},
               'debug': False,
               'pos_weight': -1},
 'type': 'ATSS'}

test_cfg = {'type': 'TestLoop'}

env_cfg = {'cudnn_benchmark': False,
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0},
 'dist_cfg': {'backend': 'nccl'}}

default_hooks = {'timer': {'type': 'IterTimerHook'}, 'logger': {'type': 'LoggerHook', 'interval': 50}}

log_processor = {'type': 'LogProcessor', 'window_size': 50, 'by_epoch': False}

launcher = 'none'

log_level = 'INFO'

test_dataloader = {'batch_size': 4,
 'dataset': {'ann_file': 'lvis_v1_val.json',
             'backend_args': None,
             'data_prefix': {'img': ''},
             'data_root': 'data/lvis/',
             'filter_cfg': {'filter_empty_gt': True, 'min_size': 32},
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1333, 800), 'type': 'Resize'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor'),
                           'type': 'PackDetInputs'}],
             'test_mode': True,
             'type': 'LVISV1Dataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'type': 'LVISMetric',
 'ann_file': 'data/lvis/lvis_v1_val.json',
 'metric': 'bbox',
 'format_only': False,
 'outfile_prefix': 'work_dirs/lvis_atss/predictions'}

work_dir = 'work_dirs/lvis_atss'
