from cmath import inf
import cv2
import numpy as np
import json
from imantics import Polygons, Mask
import BboxToolkit as bt

import math

from shapely.geometry import Polygon
from mmdet.apis import obb

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

def obb2obbox(obb):
    x, y, w, h, angle = obb
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, angle, angle+pi/2)
    theta_regular = regular_theta(theta_regular)
    obb = cv2.boxPoints(((x,y),(w_regular.item(),h_regular.item()),-180*theta_regular/pi))
    return np.int0(obb)


def draw_obb_box(frame,bbox,cat_id,color):
    x, y, w, h, angle, score = bbox
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, angle, angle+pi/2)
    theta_regular = regular_theta(theta_regular)
    obb = cv2.boxPoints(((x,y),(w_regular.item(),h_regular.item()),-180*theta_regular/pi))
    obb_box = np.int0(obb)
    cv2.drawContours(frame,[obb_box],0,color,2)
    return frame

def bbox2polygon(bbox):
    return [bbox[0],bbox[1],bbox[2],bbox[1],bbox[2],bbox[3],bbox[0],bbox[3]]

def show_obb_result(frame, result, cls_labels, score_thr = 0.3,show_bbox = True):
    
    bbox_results = result
    cat_ids = cls_labels

    for cat_bbox,cat_id in zip(bbox_results,cat_ids):
        #print(cat_bbox)
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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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

def det2polygon(det_polygon):

    polygon = []

    for i in range(int(len(det_polygon)/2)):
        polygon.append((det_polygon[i*2],det_polygon[i*2+1]))

    return Polygon(polygon)

def point_in_roi(point,roi):#roi = [x1,y1,x2,y2]
    if point[0]>roi[0] and point[0]<roi[2] and point[1]>roi[1] and point[1]<roi[3]:
        return True
    return False

def polygon_vs_roi1(polygon,roi):
    center = polygon.centroid.coords[0]
    xCenter = center[0]
    yCenter = center[1]
    return point_in_roi((xCenter,yCenter),roi)

def polygon_vs_roi2(obbox,roi):
    result = True
    for point in obbox:
        result = result and point_in_roi(point,roi)
    return result

def filt_results_with_roi(obboxes,cls_labels,score_thr = 0.3,roi = []):
    results = []
    for obbox, cls_label in zip(obboxes,cls_labels):
        #print(obb2obbox(obbox[:-1]))
        #polygon = det2polygon(obb2obbox(obbox[:-1]))
        if obbox[-1]>score_thr:# and polygon_vs_roi(polygon,roi):
            result = np.append(obb2obbox(obbox[:-1]),[obbox[-1],cls_label])
            polygon = det2polygon(result[:-2]) 
            if polygon_vs_roi1(polygon,roi): 
            #if polygon_vs_roi1(obb2obbox(obbox[:-1]),roi):  #ignore the roi
                results.append(result)
            #print(np.append(obb2obbox(obbox[:-1]),[obbox[-1],cls_label]))
    return results

def get_image_roi(height,width,scale=0.01):
    image_roi=[width*scale,height*scale,width*(1-scale),height*(1-scale)]
    return image_roi

def filt_results(obboxes,cls_labels,score_thr = 0.3):
    results = []
    for obbox, cls_label in zip(obboxes,cls_labels):
        
        if obbox[-1]>score_thr:
            results.append(np.append(obb2obbox(obbox[:-1]),[obbox[-1],cls_label]))
            #print(np.append(obb2obbox(obbox[:-1]),[obbox[-1],cls_label]))
    return results


def get_det_centers(result_dets,scale = 1):
    det_centers = []
    
    for result_det in result_dets:
        #print(((result_det[0]+result_det[4])/2,(result_det[1]+result_det[5])/2))
        #print(((result_det[2]+result_det[6])/2,(result_det[3]+result_det[7])/2))
        det_centers.append([[scale*(result_det[0]+result_det[4])/2,scale*(result_det[1]+result_det[5])/2]])
    return np.asarray(det_centers).astype(np.float32)

def get_det_edge_centers(result_dets):
    det_edge_centers = []
    det_ls = []
    for result_det in result_dets:
        edge_center = [(result_det[0]+result_det[2])/2,(result_det[1]+result_det[3])/2,
                                (result_det[2]+result_det[4])/2,(result_det[3]+result_det[5])/2,
                                (result_det[4]+result_det[6])/2,(result_det[5]+result_det[7])/2,
                                (result_det[6]+result_det[0])/2,(result_det[7]+result_det[1])/2]
        det_edge_centers.append(edge_center)
        l1 = math.sqrt((edge_center[0]-edge_center[4])+(edge_center[1]-edge_center[5]))
        l2 = math.sqrt((edge_center[2]-edge_center[6])+(edge_center[3]-edge_center[7]))
        det_ls.append([l1,l2])
    return det_edge_centers,det_ls 

def get_det_short_edget_centers(det_edge_centers,det_ls):
    pass   

def create_opencv_tracker(tracker_type='KCF'):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
        tracker_class = cv2.TrackerBoosting_create
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
        tracker_class = cv2.TrackerMIL_create
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
        tracker_class = cv2.TrackerKCF_create
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
        tracker_class = cv2.TrackerTLD_create
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
        tracker_class = cv2.TrackerMedianFlow_create
    # elif tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
        tracker_class = cv2.TrackerMOSSE_create
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
        tracker_class = cv2.TrackerCSRT_create
    else:
        assert False, "No such tracker:"+tracker_type
    return tracker

def obbox2hbbox(obbox):
    minx = float('inf')
    miny = float('inf')
    maxx = -1
    maxy = -1
    
    for i,xy in enumerate(obbox):
        if i%2==0:
            minx = xy if minx>xy else minx
            maxx = xy if maxx<xy else maxx
        else:
            miny = xy if miny>xy else miny
            maxy = xy if maxy<xy else maxy

    return [int(minx),int(miny),int(maxx-minx),int(maxy-miny)]

def obb_results2hbb_results(results):
    hbb_results = []

    for obb_result in results:

        hbb_result = [*obbox2hbbox(obb_result[:-2]),obb_result[-2],obb_result[-1]]

        hbb_results.append(hbb_result)

    return hbb_results


def init_trackers(frame,hbb_results):
    trackers = []
    other_infos = []
    for hbb_result in hbb_results:
        tracker = create_opencv_tracker()
        bbox = (hbb_result[0],hbb_result[1],hbb_result[2],hbb_result[3])
        tracker.init(frame,bbox)
        trackers.append(tracker)
        #print(hbb_result)
        other_infos.append([hbb_result[-2],hbb_result[-1]])
        
    return trackers,other_infos

def update_trackers(frame,trackers,other_infos):
    hbb_results = []
    for tracker,other_info in zip(trackers,other_infos):
        success,hbbox = tracker.update(frame)
        if success:
            hbb_result = [hbbox[0],hbbox[1],hbbox[2],hbbox[3],other_info[0],other_info[1]]
            hbb_results.append(hbb_result)
    return hbb_results

def hbbs2obbs(hbb_results):
    obb_results = []

    for hbb_result in hbb_results:
        obb_result = []
        obb_result.append(hbb_result[0])
        obb_result.append(hbb_result[1])

        obb_result.append(hbb_result[0]+hbb_result[2])
        obb_result.append(hbb_result[1])

        obb_result.append(hbb_result[0]+hbb_result[2])
        obb_result.append(hbb_result[1]+hbb_result[3])

        obb_result.append(hbb_result[0])
        obb_result.append(hbb_result[1]+hbb_result[3])

        obb_result.append(hbb_result[4])
        obb_result.append(hbb_result[5])  

        obb_results.append(obb_result)
        
    return obb_results      

def vis_hbb(frame,hbbs):
    for hbb in hbbs:
        cv2.rectangle(frame, (int(hbb[0]), int(hbb[1])), (int(hbb[0]+hbb[2]), int(hbb[1]+hbb[3])), (255, 0, 0), 2)

    return frame