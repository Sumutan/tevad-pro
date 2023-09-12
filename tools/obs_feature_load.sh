cd /home/sh/software/obsutil_linux_amd64_5.4.11

sudo -v  #获取sudo权限

export feature_name=difToken910
sudo ./obsutil cp obs://kinetics400/ucf_features/${feature_name}/ /home/sh/TEVAD/save/tmp -r -f
