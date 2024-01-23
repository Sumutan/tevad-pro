#用于从OBS上抓取特征
cd /home/sh/software/obsutil_linux_amd64_5.4.11
sudo -v  #获取sudo权限

# download folder
#sudo ./obsutil cp obs://heron-nbu/csh/ckpt /home/sh/tmp -r -f
# upload folder
#sudo ./obsutil cp "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/dataset/TAD/depth" obs://kinetics400/TAD -r -f

#Download Features
# download ucf_features
#export feature_name=UCF_9-5_9-1_finetune_DT0.25_SPa0.1   #输入特征文件夹名
#sudo ./obsutil cp obs://kinetics400/ucf_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f

# download SHT_features
#export feature_name=SHT_9-5_9-1_finetune_AISO_0.5   #输入特征文件夹名
#sudo ./obsutil cp obs://kinetics400/SHT_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f
#cd /home/sh/TEVAD/save/tmp
#sudo chmod -R 777 ${feature_name}

# download XD_Violence
#export dataset_local=/media/sh/8765-2C8C/program/dataset/XD_Violence/train/videos
#export obs_path=obs://kinetics400/XD_violence/train/
#sudo ./obsutil cp ${obs_path} ${dataset_local} -r -f
#export feature_name=XD_9-5_9-1_finetune
#sudo ./obsutil cp obs://kinetics400/XD_violence_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f


# download TAD
export feature_name=TAD_GL-20e-nofreeze
sudo ./obsutil cp obs://kinetics400/TAD_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f

#TAD_8-17_8-15_finetune_random_on_surveillance_20w_k400
#TAD_finetune_pretrain_with_RandomMaskinK400-100_457
#TAD_8-12_finetune-8-10_8-8_k400_with_surveillancel-100_469


# download UCF
#export feature_name=UCF_9-5_9-1_finetune_dif_0.5_SP_norm_a0.2
#sudo ./obsutil cp obs://kinetics400/ucf_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f

