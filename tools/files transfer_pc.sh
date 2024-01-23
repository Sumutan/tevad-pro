#用于在实验室内多台设备下进行文件传送的脚本

#ip & place for storage
export tcc3090=tcc@10.22.149.23:/media/tcc/新加卷/tcc/tccprogram/datasets/dataset/I3D_Feature_Extraction_resnet/i3d
export 4x4090=think4090@10.22.150.119:mnt/disk1/csh
#local file
export folder=/home/sh/tmp/ckpt/9-5_9-1_finetune.ckpt


#scp -r ${folder} ${tcc3090} #scp -r file tcc@10.22.149.23:/media/tcc/新加卷/csh/dataset/TAD/depth
#scp -r ${tcc3090} ${folder}
scp -r ${folder} ${4x4090}
rec_auc_abnormal : 0.5849815377110312
pr_abnormal : 0.21260509043634063
far_abnormal : 0.6598569942455327
std	mean	median	min	max	AUC
1.60	80.26	80.42	73.00	83.17
Best result:ucf-videoMae-UCF_9-1_pretrain_DT_SP-265
100%|██████████| 1000/1000 [6:39:06<00:00