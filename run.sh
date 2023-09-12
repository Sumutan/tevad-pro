echo start process!
set -e -x   # -e stop when any code raise error; -x print command

#ST default
#python main.py --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt --exp-name shanghai-i3d-official

#9.12
#ucf-videoMae-difToken910
python main.py --dataset ucf --feature-group both --fusion concat --feature-size 768 --use_dic_gt --feat-extractor videoMAE --aggregate_text --extra_loss --rgb-list list/ucf-videoMae-difToken910.list --test-rgb-list list/ucf-videoMae-test-difToken910.list --exp-name ucf-videoMae-difToken910

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
