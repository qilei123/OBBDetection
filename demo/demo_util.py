import cv2
import numpy as np
import json
from imantics import Polygons, Mask
import BboxToolkit as bt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def draw_mask(frame,cat_segm,color,alpha=0.5):
    cat_segm_array = np.array(cat_segm, dtype=bool)
    for c in range(3):
        frame[:, :, c] = np.where(
            cat_segm_array,
            frame[:, :, c] * (1 - alpha) + alpha * color[c],
            frame[:, :, c]
        )    
    return frame

def draw_bbox(frame,bbox,cat_id,color):
    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), 
        (int(bbox[2]),int(bbox[3])), color, 3)
    cv2.putText(frame, str(cat_id), (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 3, cv2.LINE_AA)
    return frame

pi = 3.141592

def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def draw_obb_box(frame,bbox,cat_id,color):
    x, y, w, h, angle, score = bbox
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, angle, angle+pi/2)
    theta_regular = regular_theta(theta_regular)
    print(bbox)
    print(((x,y),(w_regular,h_regular),theta_regular))
    obb = cv2.boxPoints(((x,y),(w_regular.item(),h_regular.item()),-180*theta_regular/pi))
    obb_box = np.int0(obb)
    cv2.drawContours(frame,[obb_box],0,color,1)
    return frame

def bbox2polygon(bbox):
    return [bbox[0],bbox[1],bbox[2],bbox[1],bbox[2],bbox[3],bbox[0],bbox[3]]

def show_obb_result(frame, result, score_thr = 0.3,show_bbox = True):

    bbox_results = result
    cat_ids = list(range(1,len(bbox_results)+1))

    for cat_bbox_results,cat_id in zip(bbox_results,cat_ids):
        for cat_bbox in cat_bbox_results:
            if cat_bbox[-1]>=score_thr:
                if show_bbox:
                    frame = draw_obb_box(frame,cat_bbox,cat_id,(255,0,0))

    return frame

def show_obbresult(frame, result, score_thr = 0.3):

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
    frame = bt.imshow_bboxes(
        frame,
        bboxes,
        labels,
        scores=scores,
        score_thr=score_thr,
        show=False)
    '''   
    for cat_bbox_results,cat_id in zip(bbox_results,cat_ids):
        for cat_bbox in cat_bbox_results:
            if cat_bbox[-1]>=score_thr:
                if show_bbox:
                    frame = draw_obb_box(frame,cat_bbox,cat_id,(255,0,0))

                    #frame = draw_mask(frame,cat_segm,(0,0,255))
    '''
    return frame

def show_result(frame, result, score_thr = 0.3,show_bbox = True, show_mask = True):

    bbox_results, segm_results = result
    cat_ids = list(range(1,len(bbox_results)+1))

    for cat_bbox_results,cat_segm_results,cat_id in zip(bbox_results, segm_results,cat_ids):
        for cat_bbox,cat_segm in zip(cat_bbox_results,cat_segm_results):
            if cat_bbox[4]>=score_thr:
                if show_bbox:
                    frame = draw_bbox(frame,cat_bbox,cat_id,(255,0,0))
                if show_mask:
                    polygons = Polygons(Mask(cat_segm).polygons())
                    frame = polygons.draw(frame,(0,0,255),2)
                    #frame = draw_mask(frame,cat_segm,(0,0,255))
    return frame

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

import csv

def json2csv(json_file_name):

    #json_file_name =  "DJI_0003 400 90 degree.avi.json"
    csv_file_name = json_file_name+".csv"

    jons_data = json.load(open(json_file_name))

    csv_file = open(csv_file_name, 'w')

    head_list = ['cat_id','trackId','frame','trackLifetime','polygon',
                'xCenter','yCenter','xVelocity','yVelocity','xAcceleration','yAcceleration',]
                #'lonVelocity','latVelocity','lonAcceleration','latAcceleration']

    write = csv.writer(csv_file)

    write.writerow(head_list)

    for key1 in jons_data:
        for item in jons_data[key1]:
            row = []
            for key2 in item:
                if key2 in head_list:
                    if key2=='trackLifetime' and item[key2]==-1:
                        row.append(0)
                    else:
                        row.append(item[key2])
            write.writerow(row)