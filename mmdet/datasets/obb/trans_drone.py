import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO

from mmdet.core import eval_arb_map, eval_arb_recalls
from ..builder import DATASETS
from ..custom import CustomDataset


@DATASETS.register_module()
class TDDataset(CustomDataset):

    CLASSES = ('Small 1-piece vehicle',
                    'Large 1-piece vehicle',
                    'Extra-large 2-piece truck',)
    coco_type = True
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        if ann_file.endswith('pkl'):
            self.coco_type = False
            ann_dict = mmcv.load(ann_file)
            contents = ann_dict['content']
            if not self.test_mode:
                data_infos = []
                for content in contents:
                    if len(content['ann']['bboxes']) != 0:
                        data_infos.append(content)
            else:
                data_infos = contents
            label_counts = [0,0,0]
            for data_info in data_infos:
                for label in data_info['ann']['labels']:
                    label_counts[int(label)-1]+=1
            print(label_counts)
            self.cat_ids = [1,2,3]
            self.cat2label = {1:0,2:1,3:2}
            return data_infos[:1000]

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()

        contents = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            img_id = info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            content = self._parse_ann_info(info,ann_info)
            contents.append(content)
        return contents
    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        if self.coco_type:
            ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
            for i, img_info in enumerate(self.data_infos):
                if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                    continue
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
            return valid_inds
        else:
            for i, img_info in enumerate(self.data_infos):
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
            return valid_inds
    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        contents = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            img_id = info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            content = self._parse_ann_info(info,ann_info)
            contents.append(content)
        return contents

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        diffs = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann['segmentation'][0])
            diffs.append(0)

        gt_masks_ann = np.array(gt_masks_ann, dtype=np.float32) if gt_masks_ann else \
                np.zeros((0, 8), dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64) if gt_labels else \
                np.zeros((0, ), dtype=np.int64)
        diffs = np.array(diffs, dtype=np.int64) if diffs else \
                np.zeros((0, ), dtype=np.int64)

        ann = dict(
            bboxes=gt_masks_ann,
            labels=gt_labels,
            diffs=diffs)

        content = dict(ann=ann)
        content.update(dict(width=img_info['width'], 
                            height=img_info['height'], 
                            filename=img_info['file_name'], 
                            id=img_info['id']))
        return content

    def format_results(self, results, save_dir=None, **kwargs):
        assert len(results) == len(self.data_infos)
        contents = []
        for result, data_info in zip(results, self.data_infos):
            info = copy.deepcopy(data_info)
            info.pop('ann')

            ann, bboxes, labels, scores = dict(), list(), list(), list()
            for i, dets in enumerate(result):
                bboxes.append(dets[:, :-1])
                scores.append(dets[:, -1])
                labels.append(np.zeros((dets.shape[0], ), dtype=np.int) + i)
            ann['bboxes'] = np.concatenate(bboxes, axis=0)
            ann['labels'] = np.concatenate(labels, axis=0)
            ann['scores'] = np.concatenate(scores, axis=0)
            info['ann'] = ann
            contents.append(info)

        if save_dir is not None:
            bt.save_pkl(save_dir, contents, self.CLASSES)
        return contents

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 iou_thr=0.5,
                 ign_diff=True,
                 scale_ranges=None,
                 use_07_metric=True,
                 proposal_nums=(100, 300, 1000)):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        if not ign_diff:
            annotations = [self.get_ann_info(i) for i in range(len(self))]
        else:
            annotations = []
            for i in range(len(self)):
                ann = self.get_ann_info(i)
                gt_bboxes = ann['bboxes']
                gt_labels = ann['labels']
                diffs = ann.get(
                    'diffs', np.zeros((gt_bboxes.shape[0], ), dtype=np.int))

                gt_ann = {}
                if ign_diff:
                    gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
                    gt_ann['labels_ignore'] = gt_labels[diffs == 1]
                    gt_bboxes = gt_bboxes[diffs == 0]
                    gt_labels = gt_labels[diffs == 0]
                gt_ann['bboxes'] = gt_bboxes
                gt_ann['labels'] = gt_labels
                annotations.append(gt_ann)

        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_arb_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=use_07_metric,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
