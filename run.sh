echo start process!
set -e -x   # -e stop when any code raise error; -x print command
#ST default
#python main.py --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --exp-name SHT-test

#i3d error
#python main.py --batch-size 64  --dataset TAD --feature-group both --fusion concat --feature-size 2048 --use_dic_gt --feat-extractor i3d --aggregate_text --extra_loss --rgb-list list/TAD-i3d.list --test-rgb-list list/TAD-test-i3d.list --exp-name TAD-i3d_batch64

python main.py --seed 0 --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.05_seed0
python main.py --seed 228 --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.05_seed228
python main.py --seed 3407 --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.05_seed3407


#TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.05/0.2
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.05
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.2.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.2.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SP_norm_batch64_a0.2

#TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_batch64_a0.2/0.4/0.8
#python main.py --dataset TAD --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.2.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.2.list --exp-name TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_batch64_a0.2
#python main.py --dataset TAD --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.4.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.4.list --exp-name TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_batch64_a0.4
python main.py --dataset TAD --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.8.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_a0.8.list --exp-name TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_MMnorm_batch64_a0.8

#ucf-segment64thon main.py
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-9-5_finetune-AISO_0.5_segment64
#python main.py --dataset ucf --seed 3407 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-9-5_finetune-AISO_0.5_segment64_seed3407

##ucf-videoMae-9-5_finetune-AISO_0.5/0.75/0.25_batch64
#python main.py --dataset ucf --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-9-5_finetune-AISO_0.5_batch64
#python main.py --dataset ucf --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken915_AISO_0.75.list --test-rgb-list list/ucf-videoMae-test-difToken915_AISO_0.75.list --exp-name ucf-videoMae-9-5_finetune-AISO_0.75_batch64
#python main.py --dataset ucf --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken918_AISO_0.25.list --test-rgb-list list/ucf-videoMae-test-difToken918_AISO_0.25.list --exp-name ucf-videoMae-9-5_finetune-AISO_0.25_batch64

#TAD-videoMae-11-1_10-15_finetune_L
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 1024 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-11-1_10-15_finetune_L.list --test-rgb-list list/TAD-videoMae-test-11-1_10-15_finetune_L.list --exp-name TAD-videoMae-11-1_10-15_finetune_L_batch64
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 1024 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-11-1_10-15_finetune_L_AISO_0.5_SP_norm.list --test-rgb-list list/TAD-videoMae-test-11-1_10-15_finetune_L_AISO_0.5_SP_norm.list --exp-name TAD-videoMae-11-1_10-15_finetune_L_AISO_0.5_SP_norm_batch64

#TAD-videoMae-9-5_finetune-AISO_0.25/0.75-batch64
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.25_SP_norm.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.25_SP_norm.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.25_SP_norm_batch64
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.75_SP_norm.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.75_SP_norm.list --exp-name TAD-videoMae-9-5_finetune-AISO_0.75_SP_norm_batch64

#TAD-videoMae-9-5_finetune-AISO_0.5-batch64
#python main.py --batch-size 64 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch64

#TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_norm
#python main.py --dataset TAD --batch-size 64 --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_norm.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP_norm.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_norm_batch64
#python main.py --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_norm.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP_norm.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_norm

#TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP
#python main.py --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-TAD_9-5_9-1_finetune_dif_0.5_SP.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_concat
#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-TAD_9-5_9-1_finetune_dif_0.5_SP.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_add
#python main.py --dataset TAD --batch-size 16 --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-TAD_9-5_9-1_finetune_dif_0.5_SP.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_batch16
#python main.py --dataset TAD --batch-size 64 --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-TAD_9-5_9-1_finetune_dif_0.5_SP.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.5_SP.list --exp-name TAD-videoMae-9-5_9-1_finetune_dif_0.5_SP_batch64


#more epochs in ucf config
#python main.py --batch-size 16 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune_concat_batch16
#python main.py --batch-size 16 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_concat_concat_batch16
#python main.py --batch-size 16 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.25.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.25.list --exp-name TAD-videoMae-9-5_finetune_AISO_0.25_concat_concat_batch16
#python main.py --batch-size 16 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.75.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.75.list --exp-name TAD-videoMae-9-5_finetune_AISO_0.75_concat_concat_batch16


#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune_add
#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_add
#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.25.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.25.list --exp-name TAD-videoMae-9-5_finetune_AISO_0.25_add
#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune_dif_0.75.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune_dif_0.75.list --exp-name TAD-videoMae-9-5_finetune_AISO_0.75_add


#python main.py --batch-size 16 --max-epoch 1500 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch32_epoch1500
#python main.py --batch-size 16 --max-epoch 1500 --seed 0 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch32_epoch1500_seed0
#python main.py --batch-size 16 --max-epoch 1500 --seed 228 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch32_epoch1500_seed228
#python main.py --batch-size 16 --max-epoch 1500 --seed 3407 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch32_epoch1500_seed3407


#TAD-videoMae-9-5_finetune-AISO_0.5-batch16
#python main.py --batch-size 16 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_batch16
#python main.py --batch-size 16 --seed 0 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed0_batch16
#python main.py --batch-size 16 --seed 228 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed228_batch16
#python main.py --batch-size 16 --seed 3407 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed3407_batch16


#11.2
#TAD-videoMae-9-5_finetune-AISO_0.5_SHTconfiig
#python main.py --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SHTconfiig
#python main.py --seed 0 --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SHTconfiig_seed0
#python main.py --seed 228 --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SHTconfiig_seed228
#python main.py --seed 3407 --dataset TAD --feature-group both --fusion add --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_SHTconfiig_seed3407


#TAD-videoMAE-9-5_finetune
#python main.py --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune
#python main.py --seed 0 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune_seed0
#python main.py --seed 228 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune_seed228
#python main.py --seed 3407 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune_seed3407


#TAD-videoMae-9-5_finetune-AISO_0.5
#python main.py --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5
#python main.py --seed 0 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed0
#python main.py --seed 228 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed228
#python main.py --seed 3407 --dataset TAD --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-AISO_0.5_seed3407


#10.28
#ucf-videoMae-test-10-13_10-4_finetune
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-10-13_10-4_finetune.list --test-rgb-list list/ucf-videoMae-test-10-13_10-4_finetune.list --exp-name ucf-videoMae-test-10-13_10-4_finetune
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-10-13_10-4_finetune.list --test-rgb-list list/ucf-videoMae-test-10-13_10-4_finetune.list --exp-name ucf-videoMae-test-10-13_10-4_finetune_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-10-13_10-4_finetune.list --test-rgb-list list/ucf-videoMae-test-10-13_10-4_finetune.list --exp-name ucf-videoMae-test-10-13_10-4_finetune_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-10-13_10-4_finetune.list --test-rgb-list list/ucf-videoMae-test-10-13_10-4_finetune.list --exp-name ucf-videoMae-test-10-13_10-4_finetune_seed3407

#10.26
#TAD-videoMae-9-5_finetune-only_vis
#python main.py --seed 0 --dataset TAD --featuren concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune-only_vis_seed0
#python main.py --seed 228 --dataset TAD --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune-only_vis_seed228
#python main.py --seed 3407 --dataset TAD --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/TAD-videoMae-9-5_9-1_finetune.list --test-rgb-list list/TAD-videoMae-test-9-5_9-1_finetune.list --exp-name TAD-videoMae-9-5_finetune-only_vis_seed3407
#python main.py --seed 0 --dataset TAD --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss  --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-only_vis-AISO_0.5_seed0
#python main.py --seed 228 --dataset TAD --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss  --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-only_vis-AISO_0.5_seed228
#python main.py --seed 3407 --dataset TAD --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss  --rgb-list list/TAD_train_list_AISO_0.5.txt --test-rgb-list list/TAD_val_list_AISO_0.5.txt --exp-name TAD-videoMae-9-5_finetune-only_vis-AISO_0.5_seed3407

#10.23 shanghai-videoMAE-9_5finetune_AISO_0.5
#python main.py --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-test-10crop.list --exp-name 9_5finetune_AISO_0.5
#python main.py --seed 0 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-test-10crop.list --exp-name 9_5finetune_AISO_0.5_seed0
#python main.py --seed 228 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-test-10crop.list --exp-name 9_5finetune_AISO_0.5_seed228
#python main.py --seed 3407 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9-5_9-1_finetune_AISO_0.5-test-10crop.list --exp-name 9_5finetune_AISO_0.5_seed3407

#10.21 shanghai-videoMAE-9_5finetune
#python main.py --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9_5finetune-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9_5finetune-test-10crop.list --exp-name shanghai-videoMAE-9_5finetune
#python main.py --seed 0 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9_5finetune-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9_5finetune-test-10crop.list --exp-name shanghai-videoMAE-9_5finetune_seed0
#python main.py --seed 228 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9_5finetune-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9_5finetune-test-10crop.list --exp-name shanghai-videoMAE-9_5finetune_seed228
#python main.py --seed 3407 --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor videoMAE --feature-size 768 --rgb-list ./list/shanghai-videoMAE-9_5finetune-train-10crop.list --test-rgb-list ./list/shanghai-videoMAE-9_5finetune-test-10crop.list --exp-name shanghai-videoMAE-9_5finetune_seed3407

#10.20
#ucf-videoMae-9-1_pretrain & ucf-videoMae-9-1_pretrain_AISO_0.5
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain.list --exp-name ucf-videoMae-9-1_pretrain
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain_AISO_0.5.list --exp-name ucf-videoMae-9-1_pretrain_AISO_0.5

#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain_AISO_0.5.list --exp-name ucf-videoMae-9-1_pretrain_AISO_0.5_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain_AISO_0.5.list --exp-name ucf-videoMae-9-1_pretrain_AISO_0.5_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain_AISO_0.5.list --exp-name ucf-videoMae-9-1_pretrain_AISO_0.5_seed3407
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain.list --exp-name ucf-videoMae-9-1_pretrain_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain.list --exp-name ucf-videoMae-9-1_pretrain_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-1_pretrain.list --test-rgb-list list/ucf-videoMae-test-9-1_pretrain.list --exp-name ucf-videoMae-9-1_pretrain_seed3407

#9.23
#ucf-videoMae-9-18_9-13_finetune_wo_loss
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-18_9-13_finetune_wo_loss.list --test-rgb-list list/ucf-videoMae-test-9-18_9-13_finetune_wo_loss.list --exp-name ucf-videoMae-9-18_9-13_finetune_wo_loss
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-18_9-13_finetune_wo_loss.list --test-rgb-list list/ucf-videoMae-test-9-18_9-13_finetune_wo_loss.list --exp-name ucf-videoMae-9-18_9-13_finetune_wo_loss_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-18_9-13_finetune_wo_loss.list --test-rgb-list list/ucf-videoMae-test-9-18_9-13_finetune_wo_loss.list --exp-name ucf-videoMae-9-18_9-13_finetune_wo_loss_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-18_9-13_finetune_wo_loss.list --test-rgb-list list/ucf-videoMae-test-9-18_9-13_finetune_wo_loss.list --exp-name ucf-videoMae-9-18_9-13_finetune_wo_loss_seed3407

#9.21
#ucf-videoMae-difToken919_AISO_0.38
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken919_AISO_0.38.list --test-rgb-list list/ucf-videoMae-test-difToken919_AISO_0.38.list --exp-name ucf-videoMae-difToken919_AISO_0.38
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken919_AISO_0.38.list --test-rgb-list list/ucf-videoMae-test-difToken919_AISO_0.38.list --exp-name ucf-videoMae-difToken919_AISO_0.38_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken919_AISO_0.38.list --test-rgb-list list/ucf-videoMae-test-difToken919_AISO_0.38.list --exp-name ucf-videoMae-difToken919_AISO_0.38_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken919_AISO_0.38.list --test-rgb-list list/ucf-videoMae-test-difToken919_AISO_0.38.list --exp-name ucf-videoMae-difToken919_AISO_0.38_seed3407

#9.20
#difToken918_AISO_0.25
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken918_AISO_0.25.list --test-rgb-list list/ucf-videoMae-test-difToken918_AISO_0.25.list --exp-name ucf-videoMae-difToken918_AISO_0.25
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken918_AISO_0.25.list --test-rgb-list list/ucf-videoMae-test-difToken918_AISO_0.25.list --exp-name ucf-videoMae-difToken918_AISO_0.25_seed5
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken918_AISO_0.25.list --test-rgb-list list/ucf-videoMae-test-difToken918_AISO_0.25.list --exp-name ucf-videoMae-difToken918_AISO_0.25_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken918_AISO_0.25.list --test-rgb-list list/ucf-videoMae-test-difToken918_AISO_0.25.list --exp-name ucf-videoMae-difToken918_AISO_0.25_seed3407

#9.17
#ucf-videoMae-difToken915_AISO_0.75
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken915_AISO_0.75.list --test-rgb-list list/ucf-videoMae-test-difToken915_AISO_0.75.list --exp-name ucf-videoMae-difToken915_AISO_0.75
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken915_AISO_0.75.list --test-rgb-list list/ucf-videoMae-test-difToken915_AISO_0.75.list --exp-name ucf-videoMae-difToken915_AISO_0.75_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken915_AISO_0.75.list --test-rgb-list list/ucf-videoMae-test-difToken915_AISO_0.75.list --exp-name ucf-videoMae-difToken915_AISO_0.75_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken915_AISO_0.75.list --test-rgb-list list/ucf-videoMae-test-difToken915_AISO_0.75.list --exp-name ucf-videoMae-difToken915_AISO_0.75_seed3407

#9.15
#ucf-videoMae-9-12_9-9_finetune_predict_the_last
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-12_9-9_finetune_predict_the_last.list --test-rgb-list list/ucf-videoMae-test-9-12_9-9_finetune_predict_the_last.list --exp-name ucf-videoMae-9-12_9-9_finetune_predict_the_last
#ucf-videoMae-difToken912_AISO_0.5_seed
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-difToken912_AISO_0.5_seed0
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-difToken912_AISO_0.5_seed228
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-difToken912_AISO_0.5_seed3407

#9.14
#ucf-videoMae-difToken912_AISO_0.5
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken912_AISO_0.5.list --test-rgb-list list/ucf-videoMae-test-difToken912_AISO_0.5.list --exp-name ucf-videoMae-difToken912_AISO_0.5

#9.13
#ucf-videoMae-difToken910
#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken910.list --test-rgb-list list/ucf-videoMae-test-difToken910.list --exp-name ucf-videoMae-difToken910_seed3407
#python main.py --seed 228 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken910.list --test-rgb-list list/ucf-videoMae-test-difToken910.list --exp-name ucf-videoMae-difToken910_seed228
#python main.py --seed 0 --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken910.list --test-rgb-list list/ucf-videoMae-test-difToken910.list --exp-name ucf-videoMae-difToken910_seed0

#9.12
#ucf-videoMae-difToken910
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken910.list --test-rgb-list list/ucf-videoMae-test-difToken910.list --exp-name ucf-videoMae-difToken910

#9.9
#ucf-videoMae-8-31_8-27_finetune_frame_only_surveillance_20w
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-31_8-27_finetune_frame_only_surveillance_20w.list --test-rgb-list list/ucf-videoMae-test-8-31_8-27_finetune_frame_only_surveillance_20w.list --exp-name ucf-videoMae-8-31_8-27_finetune_frame_only_surveillance_20w

#9.8
#ucf-videoMae-9-5_9-1_finetune
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-9-5_9-1_finetune.list --test-rgb-list list/ucf-videoMae-test-9-5_9-1_finetune.list --exp-name ucf-videoMae-9-5_9-1_finetune

#9.6
#ucf-videoMae-difToken907
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken907.list --test-rgb-list list/ucf-videoMae-test-difToken907.list --exp-name ucf-videoMae-difToken907

#9.2
#ucf-clipVis_clipText
#python main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --emb_folder text_embedding_ucf_clip --emb_dim 512 --exp-name ucf-clipVis_clipText

#9.1
#ucf-videoMaeVis_clipText
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --test-rgb-list list/ucf-videoMae-test-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --emb_folder text_embedding_ucf_clip --emb_dim 512 --exp-name ucf-videoMaeVis_clipText

#8.31
#ucf-videoMae-8-26_8-22_finetune-wo_loss
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-26_8-22_finetune-wo_loss.list --test-rgb-list list/ucf-videoMae-test-8-26_8-22_finetune-wo_loss.list --exp-name ucf-videoMae-8-26_8-22_finetune-wo_loss

#8.28
#ucf-i3dVis_clipText
#python main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --emb_dim 512 --emb_folder text_embedding_ucf_clip --exp-name ucf-i3dVis_clipText

#8.27
#ucf-videoMae-8-15_pretrain_random_on_surveillance_20w_800e-800_372_encoder
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-15_pretrain_random_on_surveillance_20w_800e-800_372_encoder.list --test-rgb-list list/ucf-videoMae-test-8-15_pretrain_random_on_surveillance_20w_800e-800_372_encoder.list --exp-name ucf-videoMae-8-15_pretrain_random_on_surveillance_20w_800e-800_372_encoder
#ucf-videoMae-2cls_finetune_50e_8_26
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-2cls_finetune_50e_8_26.list --test-rgb-list list/ucf-videoMae-test-2cls_finetune_50e_8_26.list --exp-name ucf-videoMae-2cls_finetune_50e_8_26

#8.25
##ucf-videoMae-finetune_pretrain_with_RandomMaskinK400-100_457
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune_pretrain_with_RandomMaskinK400-100_457.list --test-rgb-list list/ucf-videoMae-test-finetune_pretrain_with_RandomMaskinK400-100_457.list --exp-name ucf-videoMae-finetune_pretrain_with_RandomMaskinK400-100_457
##ucf-videoMae-8-21_8-18_finetune_k400
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-21_8-18_finetune_k400.list --test-rgb-list list/ucf-videoMae-test-8-21_8-18_finetune_k400.list --exp-name ucf-videoMae-8-21_8-18_finetune_k400

#8.23
##ucf-videoMae-test-8-17_8-15_finetune_random_on_surveillance_20w_k400
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-17_8-15_finetune_random_on_surveillance_20w_k400.list --test-rgb-list list/ucf-videoMae-test-8-17_8-15_finetune_random_on_surveillance_20w_k400.list --exp-name ucf-videoMae-test-8-17_8-15_finetune_random_on_surveillance_20w_k400
##ucf-i3d-official-only-text
#python main.py --dataset ucf --feature-group text --fusion add --aggregate_text --extra_loss --use_dic_gt --exp-name ucf-i3d-official-only-text


#8.21
#ucf-i3d-official-only-vis
#python main.py --dataset ucf --feature-group vis --fusion concat --aggregate_text --extra_loss --use_dic_gt --exp-name ucf-i3d-official-only-vis

#8.20
##ucf-videoMae-pretrain_frame_with_depth_add_loss-on_surveillance_20w
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-pretrain_frame_with_depth_add_loss-on_surveillance_20w.list --test-rgb-list list/ucf-videoMae-test-pretrain_frame_with_depth_add_loss-on_surveillance_20w.list --exp-name ucf-videoMae-pretrain_frame_with_depth_add_loss-on_surveillance_20w
##ucf-videoMae-test-8-10_8-8_pretrain_surveillance_20w_400-800e
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-10_8-8_pretrain_surveillance_20w_400-800e.list --test-rgb-list list/ucf-videoMae-test-8-10_8-8_pretrain_surveillance_20w_400-800e.list --exp-name ucf-videoMae-test-8-10_8-8_pretrain_surveillance_20w_400-800e

#8.18
#ucf-clip-MixBeforeMTN-concat
#python main_csh_MixBeforeMTN.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucf-clip-concat_before_MTN-concat
#ucf-clip-MixBeforeMTN-add
#python main_csh_MixBeforeMTN.py --dataset ucf --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucf-clip-concat_before_MTN-add
#ucf-clip-TransformerReplaceMTN-concat (error in test)
#python main_csh_TransformerMix.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucf-clip-TransformerReplaceMTN

#8.17
#ucf-videoMae-7-19_depth-800_264_encoder
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-7-19_depth-800_264_encoder --test-rgb-list list/ucf-videoMae-test-7-19_depth-800_264_encoder --exp-name ucf-videoMae-7-19_depth-800_264_encoder
#ucf-videoMae-8-12_finetune-8-10_8-8_k400_with_surveillance-100_469
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-8-12_finetune-8-10_8-8_k400_with_surveillance-100_469 --test-rgb-list list/ucf-videoMae-test-8-12_finetune-8-10_8-8_k400_with_surveillance-100_469 --exp-name ucf-videoMae-8-12_finetune-8-10_8-8_k400_with_surveillance-100_469

#8.16
# ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469_TEVAD_only_vis
#python main.py --dataset ucf --feature-group vis --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --test-rgb-list list/ucf-videoMae-test-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --exp-name ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469_TEVAD_only_vis

#8.15
##ucf-i3d-official
#python main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --exp-name ucf-i3d-official
##ucf-clip
#python main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucf-clip
##ucf-clip-only-vis
#python main.py --dataset ucf --feature-group vis --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucf-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucf-clip-only-vis

#8.14
# finetune_random_only_surveillance-100_734
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune_random_only_surveillance-100_734 --test-rgb-list list/ucf-videoMae-test-finetune_random_only_surveillance-100_734 --exp-name finetune_random_only_surveillance-100_734
# ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --test-rgb-list list/ucf-videoMae-test-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469.list --exp-name ucf-videoMae-finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469
# ucf-videoMae-finetune_frame_with_depth_401l-100_734
#python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-finetune_frame_with_depth_401l-100_734 --test-rgb-list list/ucf-videoMae-test-finetune_frame_with_depth_401l-100_734 --exp-name ucf-videoMae-finetune_frame_with_depth_401l-100_734
