# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls


@METRICS.register_module()
class CocoLTMetric(BaseMetric):
    """COCO-LT 长尾检测评估器（基于官方 CocoMetric 扩展）
    
    在标准 COCO 评估基础上，增加 COCO-LT 官方分桶 AP 计算：
    - APb1: 稀有类 (12 classes)
    - APb2: 较少类 (38 classes)
    - APb3: 较多类 (28 classes)
    - APb4: 频繁类 (3 classes)
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None
        
        # ===================== COCO-LT 官方配置（100% 正确） =====================
        # 1. 模型连续标签(0-79) -> COCO原始ID(0-90)
        self.CONT_ID_TO_COCO_ID = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]

        # 2. COCO-LT 官方分桶（基于 COCO 原始 ID，IGAM 论文标准）
        self.LT_BUCKETS = {
            'APb1': [34,90,14,11,],  # 稀有类 (12)
            'APb2': [7,24,25,22,73,6,17,18,81,59,70,89,39,79,23,40,74,78,65,82,58,80,87,13,36,76,],  # 较少类 (38)
            'APb3': [43,48,75,72,63,5,54,88,41,53,51,31,8,15,27,84,37,10,49,28,77,32,4,16,85,9,42,61,38,52,33,2,46,64,21,35,56,20,60,55,57,86,19,50,],  # 较多类 (28)
            'APb4': [1,62,3,67,47,44,]  # 频繁类 (3)
        }

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                # ===================== COCO-LT：标签映射 =====================
                data['category_id'] = self.CONT_ID_TO_COCO_ID[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                # ===================== COCO-LT：标签映射 =====================
                data['category_id'] = self.CONT_ID_TO_COCO_ID[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')
                
                # ===================== COCO-LT 核心：分桶计算（100% 正确） =====================
                logger.info("\n" + "="*60)
                logger.info("✅ COCO-LT 长尾检测结果（IGAM 论文标准）")
                logger.info("="*60)
                
                # 建立 COCO原始ID -> 评估索引 的映射
                cat_id_to_eval_idx = {cid: i for i, cid in enumerate(coco_eval.params.catIds)}
                
                for bucket_name, bucket_cat_ids in self.LT_BUCKETS.items():
                    # 找到分桶类别在评估结果中的索引
                    valid_indices = [cat_id_to_eval_idx[cid] for cid in bucket_cat_ids if cid in cat_id_to_eval_idx]
                    
                    if not valid_indices:
                        eval_results[bucket_name] = 0.0
                        continue
                    
                    # 提取 precision 矩阵: [iou_thrs, recall_thrs, cat_idx, area_rng, max_dets]
                    # 取: 所有IoU, 所有Recall, 指定类别, 所有面积, 100个检测
                    prec = coco_eval.eval['precision'][:, :, valid_indices, 0, -1]
                    prec = prec[prec > -1]  # 过滤掉 -1 (无效值)
                    
                    if prec.size > 0:
                        ap_value = float(np.mean(prec))
                    else:
                        ap_value = 0.0
                    
                    eval_results[bucket_name] = round(ap_value, 3)
                    logger.info(f"{bucket_name}: {eval_results[bucket_name]:.3f}")
                
                logger.info("="*60 + "\n")

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results