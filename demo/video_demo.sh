# python demo/video_demo.py \
#     configs/obb/app/trans_drone/fcos_obb_r50_fpn_gn-head_4x4_1x_td_patch.py \
#     data/td/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_td_patch/latest.pth \
#     --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
#     --video_dir /home/qilei/DATASETS/trans_drone/trans_drone_videos2/DJI_0063.MOV \
#     --out_dir /home/qilei/DATASETS/trans_drone/DJI_0063_1.MOV --mix

# python demo/video_demo.py \
#     configs/obb/app/vis_drone/fcos_obb_r50_fpn_gn-head_4x4_1x_vd.py \
#     data/vd/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_vd/latest.pth \
#     --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
#     --video_dir /home/qilei/DATASETS/trans_drone/trans_drone_videos2/DJI_0063.MOV \
#     --out_dir /home/qilei/DATASETS/trans_drone/DJI_0063_vd.MOV --mix --save_imgs

# export CUDA_VISIBLE_DEVICES=1
# python demo/video_demo.py \
#     configs/obb/app/vis_drone/fcos_obb_r50_fpn_gn-head_4x4_1x_vdr.py \
#     data/vdr/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_vdr_rotate/latest.pth \
#     --video_dir "/home/qilei/DATASETS/trans_drone/trans_drone_videos/DJI_0004 Thermal 400.MOV" \
#     --out_dir "/home/qilei/DATASETS/trans_drone/DJI_0004 Thermal 400_rotate.MOV" --save_imgs #--mix
#     #--split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \

# python demo/video_demo.py \
#     configs/obb/app/trans_drone/fcos_obb_r50_fpn_gn-head_4x4_1x_td_mixpatch.py \
#     data/td/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_td_mixpatch/latest.pth \
#     --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
#     --video_dir /home/qilei/DATASETS/trans_drone/trans_drone_videos2/DJI_0063.MOV \
#     --out_dir /home/qilei/DATASETS/trans_drone/DJI_0063_mixpatch.MOV --mix --save_imgs

# export CUDA_VISIBLE_DEVICES=1
# python demo/video_demo.py \
#     configs/obb/app/trans_drone/fcos_obb_r50_fpn_gn-head_4x4_1x_td_mixmorepatch.py \
#     data/td/work_dirs/fcos_obb_r50_fpn_gn-head_4x4_1x_td_mixmorepatch/latest.pth \
#     --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
#     --video_dir /home/qilei/DATASETS/trans_drone/trans_drone_videos2/DJI_0063.MOV \
#     --out_dir /home/qilei/DATASETS/trans_drone/DJI_0063_mixmorepatch.MOV --mix --save_imgs

# export CUDA_VISIBLE_DEVICES=1
# export CONFIG=fcos_obb_r50_fpn_gn-head_4x4_1x_td_patch
# export PARAM_FOLDER=${CONFIG}_rotate
# export VIDEO_DIR=/home/qilei/DATASETS/trans_drone/trans_drone_videos2
# files=$(ls $VIDEO_DIR)
# for VIDEO_NAME in $files
# do
#     echo $VIDEO_NAME
#     python demo/video_demo.py \
#         configs/obb/app/trans_drone/$CONFIG.py \
#         data/td/work_dirs/${PARAM_FOLDER}/latest.pth \
#         --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
#         --video_dir "${VIDEO_DIR}/${VIDEO_NAME}" \
#         --out_dir "/home/qilei/DATASETS/trans_drone/andover_worster/work_dirs/${PARAM_FOLDER}/video_results/${VIDEO_NAME}" --mix --save_imgs
# done

export CUDA_VISIBLE_DEVICES=1
export CONFIG=fcos_obb_r50_fpn_gn-head_4x4_1x_td_patch
export PARAM_FOLDER=${CONFIG}_rotate
export VIDEO_DIR=/home/qilei/DATASETS/trans_drone/trans_drone_videos
files=$(ls $VIDEO_DIR|tr " " "_")

for VIDEO_NAME in $files
do
    echo $VIDEO_NAME
    NEW_VIDEO_NAME=$(tr "_" " " <<<$VIDEO_NAME)
    echo $NEW_VIDEO_NAME
    #python demo/video_demo.py \
    #    configs/obb/app/trans_drone/$CONFIG.py \
    #    data/td/work_dirs/${PARAM_FOLDER}/latest.pth \
    #    --split BboxToolkit/tools/split_configs/trans_drone/aw_test.json \
    #    --video_dir "${VIDEO_DIR}/${VIDEO_NAME}" \
    #    --out_dir "/home/qilei/DATASETS/trans_drone/andover_worster/work_dirs/${PARAM_FOLDER}/video_results/${VIDEO_NAME}" --mix --save_imgs
done

