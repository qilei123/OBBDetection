import argparse

import cv2
import torch
import os
from mmdet.apis import inference_detector, init_detector
from mmdet.apis import inference_detector_huge_image,fast_inference_detector_huge_image
from demo_util import *
from Tracks import *

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
    
    tmer = tracks_manager()

    frame_number = 0
    show_fq = 30
    
    if frame_number>0:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
    
    image_roi = get_image_roi(img.shape[0],img.shape[1],scale=0.002)

    tmer.set_img_roi(image_roi)

    while ret_val:
        hbb_results = []
        if frame_number%5==0:

            if not args.split:
                result = inference_detector(model, img)
            else:
                nms_cfg = dict(type='BT_nms', iou_thr=0.1)
                result = inference_detector_huge_image(model,img,args.split,nms_cfg,args.mix)

            #img = show_obb_result(img,*result)


            #results = filt_results(*result)

            #this two line get rid of the object near the edges of the image
            
            results = filt_results_with_roi(*result,roi=image_roi)

            hbb_results = obb_results2hbb_results(results)

            multi_trackers,other_infos = init_trackers(img,hbb_results)

        else:

            hbb_results = update_trackers(img,multi_trackers,other_infos)

        obb_results = hbbs2obbs(hbb_results)

        tmer.update_with_obbox(hbb_results,frame_number)
        img = tmer.vis(img)
        if args.save_imgs and frame_number%show_fq==0:
            if not os.path.exists(args.out_dir[:-4]):
                os.makedirs(args.out_dir[:-4])
            save_img_dir = os.path.join(args.out_dir[:-4],str(frame_number).zfill(10)+".jpg")
            cv2.imwrite(save_img_dir,img)

        if video_writer:
            video_writer.write(img)
        
        ret_val, img = video_reader.read()
        frame_number+=1
        #this is for debug
        #if frame_number==450:
        #    break

    if isinstance( args.out_dir,str):
        tmer.save_results(args.out_dir+".json")
        json2csv(args.out_dir+".json")
if __name__ == '__main__':
    main()