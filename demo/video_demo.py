import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.apis import inference_detector_huge_image

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
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
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
    while ret_val:
        
        if not args.split:
            result = inference_detector(model, img)
        else:
            nms_cfg = dict(type='BT_nms', iou_thr=0.5)
            result,windows = inference_detector_huge_image(model,img,args.split,nms_cfg)
            print(windows)
        img = model.show_result(img, result, show=False)
        if video_writer:
            video_writer.write(img)

        ret_val, img = video_reader.read()

if __name__ == '__main__':
    main()