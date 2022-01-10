import argparse

import cv2
import torch
import os
from mmdet.apis import inference_detector, init_detector
from mmdet.apis import inference_detector_huge_image,fast_inference_detector_huge_image
from demo_util import *
from Tracks import *
from draw_util import *

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--split', type=str, default='', help='split configs in BboxToolkit/tools/split_configs')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--video_dir', type=str, default='', help='video_dir')
    parser.add_argument(
        '--out_dir', type=str, default='', help='out_dir')
    parser.add_argument(
        '--save_imgs',
        action='store_true',
        help='whether to save images every second 1 frame as fps is 30')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--mix',
        action='store_true',
        help='whether to mix patchs and the whole image detection')
    args = parser.parse_args()
    return args


def main():

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    video_reader = cv2.VideoCapture(args.video_dir)
    video_writer = None
    if args.out_dir:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out_dir, fourcc, video_reader.get(cv2.CAP_PROP_FPS),
            (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    ret_val, img = video_reader.read()
    
    pre_gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    curr_gray_img = None

    tmer = tracks_manager()

    frame_number = 0
    if frame_number>0:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
    while ret_val:
        
        if not args.split:
            result = inference_detector(model, img)
        else:
            nms_cfg = dict(type='BT_nms', iou_thr=0.1)
            result = inference_detector_huge_image(model,img,args.split,nms_cfg,args.mix)

        img = show_obb_result(img,*result)
        
        image_roi = get_image_roi(img.shape[0],img.shape[1])

        drawrect(img,(image_roi[0],image_roi[1]),(image_roi[2],image_roi[3]),(255,255,0),2,'dotted')

        results = filt_results_with_roi(*result,roi=image_roi)

        result_centers = get_det_centers(results)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(gray_img, mask = None,
							**feature_params)

        print('p0')
        print(p0)
        print('result_centers')
        print(result_centers)
        #results = filt_results(*result)
        tmer.update_with_obbox(results,frame_number)
        img = tmer.vis(img)
        if args.save_imgs and frame_number%30==0:
            if not os.path.exists(args.out_dir[:-4]):
                os.makedirs(args.out_dir[:-4])
            save_img_dir = os.path.join(args.out_dir[:-4],str(frame_number).zfill(10)+".jpg")
            cv2.imwrite(save_img_dir,img)

        if video_writer:
            video_writer.write(img)
        
        ret_val, img = video_reader.read()

        curr_gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        frame_number+=1

    if isinstance( args.out_dir,str):
        tmer.save_results(args.out_dir+".json")
        json2csv(args.out_dir+".json")
if __name__ == '__main__':
    main()