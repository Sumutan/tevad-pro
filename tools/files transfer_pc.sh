#!/bin/bash
#用于在实验室内多台设备下进行文件传送的脚本

#ip & place for storage
export tcc3090=tcc@10.22.149.23:/media/tcc/新加卷/tcc/tccprogram/datasets/dataset/I3D_Feature_Extraction_resnet/i3d
export surver4x4090=think4090@10.22.150.119 #CVPRS518
export surver2080Ti=sh@10.22.149.3:/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/tube_wc3090/3_22
export surver4xA6000=think@10.33.11.31:/mnt/disk2/tcc/anomly_feature.pytorch-main
export lin3090=think@192.168.1.88:/media/think/1432FCCA32FCB23A/tcc/LAP/save/TAD/10crop_clip_large
export cy518=cy518@10.22.149.25:/home/cy518/program/yolov5-master/video/video_dataset.zip
#local file
export folder=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/TAD_10crop_clip/

#export aim=think4090@10.22.150.119:/mnt/disk2/tcc/dataset/anomly_feature.pytorch-main/tmp/zuijiu_v1/output_all_filter_3s_l1
#export src=/media/cw/584485FC4485DD5E/csh/dataset/video_dataset/output_all_filter_3s/output_all_filter_3s_l1
#scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}


#download
#export src=cy518@10.22.149.25:/home/cy518/program/yolo_tracking-3/zuijiu_video_dataset_v2
#export aim=/media/cw/584485FC4485DD5E/csh/dataset/zuijiu/zuijiu_video_dataset_v2
##scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#export src=cy518@10.22.149.25:/home/cy518/program/yolov5-master/video/output_all_filter_3s
#export aim=/media/cw/584485FC4485DD5E/csh/dataset/video_dataset/output_all_filter_3s
##scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#upload
#export src=/media/cw/584485FC4485DD5E/csh/anomly_feature.pytorch-main/I3D_infer_pack.zip
#export aim=cy518@10.22.149.25:/home/cy518/program/yolo_tracking-3
##scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#2080
#export date=4_6
#export src=sh@10.22.149.3:/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/tube_wc3090/${date}
#export aim=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/tmp/from2080/${date}
##echo "123456" | scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#export date=4_5
#export src=think@10.33.11.31:/home/think/tcc/10crop_clip_large
#export aim=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/tmp/fromA6000/${date}
##echo "123456" | scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}


#export date=4_9
  #export src=think4090@10.22.150.119:/mnt/disk2/tcc/dataset/anomly_feature.pytorch-main
#export aim=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/tmp/from_ALL/${date}
##echo "123456" | scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#export date=4_12
#export src=tcc@10.22.149.23:/media/tcc/新加卷/tcc/tccprogram/UR-DMU/dataset/ucfcrime
#export aim=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/tmp/from_ALL/${date}
##echo "123456" | scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}


#export src=cy518@10.22.149.25:/home/cy518/program/yolo_tracking-3/video/output
#export aim=/media/cw/584485FC4485DD5E/csh/anomly_feature.pytorch-main/finetune/zuijiu_i3d_finetune/dataset/tmp
##scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#export src=/media/cw/584485FC4485DD5E/csh/anomly_feature.pytorch-main/finetune/zuijiu_i3d_finetune/dataset/zuijiu_video_dataset_v3.3/normal_video
#export aim=think@10.61.62.149:/mnt/hjl/code/
##scp -r ${src} ${aim}
#rsync -rav --progress ${src} ${aim}

#export src=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/Crime/UCF_FBseparation_threthold10-videoMAE-ST
#export src=/media/cw/584485FC4485DD5E/csh/tevad-pro/save/TAD/TAD_FBseparation_threthold10-videoMAE-ST


# A6000 GV Captinons
#export src=/media/cw/584485FC4485DD5E/csh/VideoGeneration/GeneratedVideos/descriptions_1
#export aim=think@10.61.62.178:/home/think/ppp/VideoLLaMA2/datasets/videos_csh/GeneratedVideos
#rsync -rav --progress ${src} ${aim}

# 4090 LAP

export src="/home/cw/sh/dataset/Crime/feature/8-26_8-22_SVMAE_wo-loss_CLIP-B"
export aim="${surver4x4090}:/mnt/disk2/csh/download"
#export aim="think@10.118.245.183:/home/think/Downloads/features"
rsync -rav --progress ${src}  ${aim}



#export surver4xA6000=think@10.33.11.31:/mnt/disk2/tcc/anomly_feature.pytorch-main