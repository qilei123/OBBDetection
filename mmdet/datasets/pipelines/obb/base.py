import re
import BboxToolkit as bt

import cv2
import mmcv
import warnings
import itertools
import numpy as np
import pycocotools.mask as maskUtils

from mmcv.parallel import DataContainer as DC
from mmdet.core import PolygonMasks, BitmapMasks
from mmdet.datasets.builder import PIPELINES

from ..loading import LoadAnnotations
from ..formating import DefaultFormatBundle, Collect, to_tensor
from ..transforms import RandomFlip
from ..compose import Compose


def vis(results,save_dir):
    img = np.array(results['img'])
    for box in results['ann_info']['bboxes']:
        
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=1)
        cv2.line(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), (0, 0, 255), thickness=1)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 0, 255), thickness=1)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[0]), int(box[1])), (0, 0, 255), thickness=1)
    cv2.imwrite(save_dir,img)

pi = 3.141592

def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def draw_obb_box(frame,bbox,color):
    x, y, w, h, angle = bbox
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, angle, angle+pi/2)
    theta_regular = regular_theta(theta_regular)
    obb = cv2.boxPoints(((x,y),(w_regular.item(),h_regular.item()),-180*theta_regular/pi))
    obb_box = np.int0(obb)
    cv2.drawContours(frame,[obb_box],0,color,2)
    return frame

def vis_obb(results,save_dir):
    img = np.array(results['img'])
    for obbox in results['gt_obboxes']:
        draw_obb_box(img,obbox,(0, 0, 255))
    cv2.imwrite(save_dir,img)    

def mask2obb(gt_masks):
    obboxes = []
    if isinstance(gt_masks, PolygonMasks):
        for mask in gt_masks.masks:
            all_mask_points = np.concatenate(mask, axis=0)[None, ...]
            obboxes.append(bt.bbox2type(all_mask_points, 'obb'))
    elif isinstance(gt_masks, BitmapMasks):
        for mask in gt_masks.masks:
            try:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            except ValueError:
                _, contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_contour = max(contours, key=len).reshape(1, -1)
            obboxes.append(bt.bbox2type(max_contour, 'obb'))
    else:
        raise NotImplementedError

    if not obboxes:
        return np.zeros((0, 5), dtype=np.float32)
    else:
        obboxes = np.concatenate(obboxes, axis=0)
        return obboxes


def mask2poly(gt_masks):
    polys = []
    if isinstance(gt_masks, PolygonMasks):
        for mask in gt_masks.masks:
            if len(mask) == 1 and mask[0].size == 8:
                polys.append(mask)
            else:
                all_mask_points = np.concatenate(mask, axis=0)[None, ...]
                obbox = bt.bbox2type(all_mask_points, 'obb')
                polys.append(bt.bbox2type(obbox, 'poly'))
    elif isinstance(gt_masks, BitmapMasks):
        for mask in gt_masks.masks:
            try:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            except ValueError:
                _, contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_contour = max(contours, key=len).reshape(1, -1)
            obbox = bt.bbox2type(max_contour, 'obb')
            polys.append(bt.bbox2type(obbox, 'poly'))
    else:
        raise NotImplementedError

    if not polys:
        return np.zeros((0, 8), dtype=np.float32)
    else:
        polys = np.concatenate(polys, axis=0)
        return polys


def poly2mask(polys, w, h, mask_type='polygon'):
    assert mask_type in ['polygon', 'bitmap']
    if mask_type == 'bitmap':
        masks = []
        for poly in polys:
            rles = maskUtils.frPyObjects([poly.tolist()], h, w)
            masks.append(maskUtils.decode(rles[0]))
        gt_masks = BitmapMasks(masks, h, w)

    else:
        gt_masks = PolygonMasks([[poly] for poly in polys], h, w)
    return gt_masks


@PIPELINES.register_module()
class FliterEmpty:

    def __call__(self, results):
        num_objs = 0
        for k in ['gt_bboxes', 'gt_masks', 'gt_labels']:
            if k in results:
                num_objs += len(results[k])
        if num_objs == 0:
            return None

        return results


@PIPELINES.register_module()
class LoadOBBAnnotations(LoadAnnotations):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_seg=False,
                 with_poly_as_mask=True,
                 poly2mask=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_poly_as_mask = with_poly_as_mask
        self.with_label = with_label
        self.with_mask = False
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        gt_bboxes = ann_info['bboxes'].copy()
        results['gt_bboxes'] = bt.bbox2type(gt_bboxes, 'hbb')
        results['bbox_fields'].append('gt_bboxes')

        if self.with_poly_as_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            polys = bt.bbox2type(gt_bboxes.copy(), 'poly')
            mask_type = 'bitmap' if self.poly2mask else 'polygon'
            gt_masks = poly2mask(polys, w, h, mask_type)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        return results


@PIPELINES.register_module()
class OBBRandomFlip(RandomFlip):

    def __init__(self, h_flip_ratio=None, v_flip_ratio=None):
        if h_flip_ratio is not None:
            assert h_flip_ratio >= 0 and h_flip_ratio <= 1
        if v_flip_ratio is not None:
            assert v_flip_ratio >= 0 and v_flip_ratio <= 1

        self.h_flip_ratio = h_flip_ratio
        self.v_flip_ratio = v_flip_ratio

    def __call__(self, results):
        if 'flip' in results:
            if 'flip_direction' in results:
                direction = results['flip_direction']
                results['h_flip'] = results['flip'] \
                        if direction == 'horizontal' else False
                results['v_flip'] = results['flip'] \
                        if direction == 'vertical' else False
            else:
                results['h_flip'] = results['flip']
                results['v_flip'] = False

        if 'h_flip' not in results:
            h_flip = True if np.random.rand() < self.h_flip_ratio else False
            results['h_flip'] = h_flip
        if 'v_flip' not in results:
            v_flip = True if np.random.rand() < self.v_flip_ratio else False
            results['v_flip'] = v_flip

        if results['h_flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction='horizontal')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              'horizontal')
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip('horizontal')

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction='horizontal')

        if results['v_flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction='vertical')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              'vertical')
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip('vertical')

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction='vertical')
        return results


@PIPELINES.register_module()
class Mask2OBB(object):

    def __init__(self,
                 mask_keys=['gt_masks', 'gt_masks_ignore'],
                 obb_keys=['gt_obboxes', 'gt_obboxes_ignore'],
                 obb_type='obb'):
        assert len(mask_keys) == len(obb_keys)
        assert obb_type in ['obb', 'poly']
        self.mask_keys = mask_keys
        self.obb_keys = obb_keys
        self.obb_type = obb_type

    def __call__(self, results):
        trans_func = mask2obb if self.obb_type == 'obb' else mask2poly
        for mask_k, obb_k in zip(self.mask_keys, self.obb_keys):
            if mask_k in results:
                mask = results[mask_k]
                obb = trans_func(mask)
                results[obb_k] = obb
        return results


@PIPELINES.register_module()
class OBBDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore',
                    'gt_obboxes', 'gt_obboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            if key in ['gt_obboxes'] + results.get('bbox_fields', []):
                results[key] = results[key].astype(np.float32)
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results


@PIPELINES.register_module()
class OBBCollect(Collect):

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'h_flip', 'v_flip', 'angle',
                            'matrix', 'rotate_after_flip', 'img_norm_cfg')):
        super(OBBCollect, self).__init__(keys, meta_keys)
        


@PIPELINES.register_module()
class RandomOBBRotate(object):

    def __init__(self,
                 rotate_after_flip,
                 angles=(0, 90),
                 rotate_mode='range',
                 vert_rate=0.5,
                 vert_cls=None,
                 keep_shape=True,
                 keep_iof_thr=0.7):
        assert rotate_mode in ['range', 'value']
        if rotate_mode == 'range':
            assert len(angles) == 2
        self.rotate_after_flip = rotate_after_flip
        self.angles = angles
        self.rotate_mode = rotate_mode
        self.vert_rate = vert_rate
        self.vert_cls = vert_cls
        self.keep_shape = keep_shape
        self.keep_iof_thr = keep_iof_thr

    def get_random_angle(self, results):
        vert = False
        if self.vert_cls is not None:
            if 'cls' not in results:
                raise ValueError(
                    'need class order when vert_cls is not None')
            vert_lbls, cls_list = [], results['cls']
            for c in self.vert_cls:
                if c in cls_list:
                    vert_lbls.append(cls_list.index(c))
            if 'gt_labels' in results:
                labels = results['gt_labels']
                for i in vert_lbls:
                    if (labels == i).any():
                        vert = True
        vert = True if np.random.rand() < self.vert_rate else vert

        if vert:
            angles = [a for a in [-90, 0, 90, 180]
                      if a >= min(self.angles) and a <= max(self.angles)]
            angles = angles + [0] if 0 not in angles else angles
            np.random.shuffle(angles)
            angle = angles[0]
        else:
            if self.rotate_mode == 'value':
                angles = list(self.angles)
                angles = angles + [0] if 0 not in angles else angles
                np.random.shuffle(angles)
                angle = angles[0]
            else:
                angle_min, angle_max = min(self.angles), max(self.angles)
                angle = (angle_max - angle_min) * np.random.rand() + angle_min
        return angle

    def get_matrix_and_size(self, results):
        angle = results['angle']
        height, width = results['img_shape'][:2]
        if self.keep_shape:
            center = ((width - 1) * 0.5, (height - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(center, angle, 1)
        else:
            matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
            img_bbox = np.array([[0, 0, width, 0, width, height, 0, width]])
            img_bbox = bt.bbox2type(bt.warp(img_bbox, matrix), 'hbb')

            width = int(img_bbox[0, 2] - img_bbox[0, 0] + 1)
            height = int(img_bbox[0, 3] - img_bbox[0, 1] + 1)
            matrix[0, 2] = -img_bbox[0, 0]
            matrix[1, 2] = -img_bbox[0, 1]
        return matrix, width, height

    def base_rotate(self, results, matrix, w, h, img_bound):
        if 'img' in results:
            img = cv2.warpAffine(results['img'], matrix, (w, h))
            results['img'] = img
            results['img_shape'] = img.shape

        if 'gt_masks' in results:
            polys = mask2poly(results['gt_masks'])
            warped_polys = bt.warp(polys, matrix)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_polys, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_polys = warped_polys[if_inwindow]

            if isinstance(results['gt_masks'], BitmapMasks):
                results['gt_masks'] = poly2mask(warped_polys, w, h, 'bitmap')
            elif isinstance(results['gt_masks'], PolygonMasks):
                results['gt_masks'] = poly2mask(warped_polys, w, h, 'polygon')
            else:
                raise NotImplementedError

            if 'gt_bboxes' in results:
                results['gt_bboxes'] = bt.bbox2type(warped_polys, 'hbb')

        elif 'gt_bboxes' in results:
            warped_bboxes = bt.warp(results['gt_bboxes'], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_bboxes = warped_bboxes[if_inwindow]
            results['gt_bboxes'] = warped_bboxes

        if 'gt_labels' in results and self.keep_shape:
            results['gt_labels'] = results['gt_labels'][if_inwindow]

        for k in results.get('aligned_fields', []):
            if self.keep_shape:
                results[k] = results[k][if_inwindow]

        # return False

    def __call__(self, results):
        results['rotate_after_flip'] = self.rotate_after_flip
        if 'angle' not in results:
            results['angle'] = self.get_random_angle(results)
        if results['angle'] == 0:
            results['matrix'] = np.eye(3)
            return results
        matrix, w, h = self.get_matrix_and_size(results)
        results['matrix'] = matrix
        img_bound = np.array([[0, 0, w, 0, w, h, 0, h]])
        self.base_rotate(results, matrix, w, h, img_bound)

        for k in results.get('img_fields', []):
            if k != 'img':
                results[k] = cv2.warpAffine(results[k], matrix, (w, h))

        for k in results.get('bbox_fields', []):
            if k == 'gt_bboxes':
                continue
            warped_bboxes = bt.warp(results[k], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                warped_bboxes = warped_bboxes[iofs[:, 0] > self.keep_iof_thr]
            results[k] = warped_bboxes

        for k in results.get('mask_fields', []):
            if k == 'gt_masks':
                continue
            polys = mask2poly(results[k])
            warped_polys = bt.warp(polys, matrix)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_polys, img_bound, mode='iof')
                warped_polys = warped_polys[iofs[:, 0] > self.keep_iof_thr]
            if isinstance(results[k], BitmapMasks):
                results[k] = poly2mask(warped_polys, w, h, 'bitmap')
            elif isinstance(results[k], PolygonMasks):
                results[k] = poly2mask(warped_polys, w, h, 'polygon')
            else:
                raise NotImplementedError

        for k in results.get('seg_fields', []):
            results[k] = cv2.warpAffine(results[k], matrix, (w, h))

        return results


@PIPELINES.register_module()
class MultiScaleFlipRotateAug(object):

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 h_flip=False,
                 v_flip=False,
                 rotate=False):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be setted')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotate = rotate

    def __call__(self, results):
        aug_data = []
        aug_cfgs = [[False, False, 0]]
        if self.h_flip:
            aug_cfgs.append([True, False, 0])
        if self.v_flip:
            aug_cfgs.append([False, True, 0])
        if self.rotate:
            aug_cfgs.append([False, False, 90])
        for scale in self.img_scale:
            for h_flip, v_flip, angle in aug_cfgs:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['h_flip'] = h_flip
                _results['v_flip'] = v_flip
                _results['angle'] = angle
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'h_flip={self.h_flip}, v_flip={self.v_flip}, '
        repr_str += f'angle={self.rotate})'
        return repr_str
