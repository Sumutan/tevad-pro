#用于在实验室内多台设备下进行文件传送的脚本

#ip & place for storage
export tcc3090=tcc@10.22.149.23:/media/tcc/新加卷/csh/tmp
export tcc3090_password=123
#local file
export folder=/home/sh/TEVAD/save/tmp/9-5_9-1_finetune_AISO_0.5_1crop


scp -r ${folder} ${tcc3090} #scp -r file tcc@10.22.149.23:/media/tcc/新加卷/csh/tmp
echo ${tcc3090_password}