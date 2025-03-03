echo start process!
set -x # -e stop when any code raise error; -x print command
start_time=$(date +%s)

#python main.py --seed 0 --dataset violence --feature-group vis --fusion concat --feature-size 512 --use_dic_gt --feat-extractor clip --aggregate_text --extra_loss --rgb-list list/violence-CLIP-10crop_clip_B.list --test-rgb-list list/violence-CLIP-test-10crop_clip_B.list --exp-name violence-CLIP-10crop_clip_B_onlyvis_seed0

#python main.py --seed 3407 --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name test
#python main.py --seed 0 --dataset violence --feature-group vis --fusion concat --feature-size 512 --use_dic_gt --feat-extractor clip --aggregate_text --extra_loss --rgb-list list/violence-CLIP-10crop_clip_B-g2.list --test-rgb-list list/violence-CLIP-test-10crop_clip_B.list --exp-name violenceg2-CLIP-10crop_clip_B_seed0_vis


# 1.15 XD and clip (only vis)
#python main.py --seed 0 --dataset violence --feature-group both --fusion concat --feature-size 512 --use_dic_gt --feat-extractor clip --aggregate_text --extra_loss --rgb-list list/violence-CLIP-10crop_clip_B.list --test-rgb-list list/violence-CLIP-test-10crop_clip_B.list --exp-name violence-CLIP-10crop_clip_B_seed0
#python main.py --seed 228 --dataset violence --feature-group both --fusion concat --feature-size 512 --use_dic_gt --feat-extractor clip --aggregate_text --extra_loss --rgb-list list/violence-CLIP-10crop_clip_B.list --test-rgb-list list/violence-CLIP-test-10crop_clip_B.list --exp-name violence-CLIP-10crop_clip_B_seed228
#python main.py --seed 3407 --dataset violence --feature-group both --fusion concat --feature-size 512 --use_dic_gt --feat-extractor clip --aggregate_text --extra_loss --rgb-list list/violence-CLIP-10crop_clip_B.list --test-rgb-list list/violence-CLIP-test-10crop_clip_B.list --exp-name violence-CLIP-10crop_clip_B_seed3407

# 1.15 ucfg2 and clip (only vis)
#python main.py --seed 3407 --dataset ucfg2 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg2-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucfg2-clip-only_vis_seed3407
#python main.py --seed 228 --dataset ucfg2 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg2-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucfg2-clip-only_vis_seed228
#python main.py --seed 0 --dataset ucfg2 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg2-clip.list --test-rgb-list ./list/ucf-clip-test.list --exp-name ucfg2-clip-only_vis_seed0

#12.21 验证数据不足假设
#python main.py --seed 3407 --dataset ucf25% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-25%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-25%_seed3407
#python main.py --seed 3407 --dataset ucf50% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-50%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-50%_seed3407
#python main.py --seed 3407 --dataset ucf75% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-75%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-75%_seed3407

#python main.py --seed 228 --dataset ucf25% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-25%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-25%_seed228
#python main.py --seed 228 --dataset ucf50% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-50%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-50%_seed228
#python main.py --seed 228 --dataset ucf75% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-75%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-75%_seed228

#python main.py --seed 0 --dataset ucf25% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-25%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-25%_seed0
#python main.py --seed 0 --dataset ucf50% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-50%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-50%_seed0
#python main.py --seed 0 --dataset ucf75% --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/GV/ucf-clip-75%.list --test-rgb-list ./list/GV/ucf-clip-test.list --exp-name ucf-clip-75%_seed0


#12.20
#python main.py --seed 3407 --dataset ucfg1 --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip_seed3407
#python main.py --seed 228 --dataset ucfg1 --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip_seed228
#python main.py --seed 0 --dataset ucfg1 --feature-group both --fusion concat --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip_seed0

# 12.19 ucfg1 and clip (only vis)
#python main.py --dataset ucfg1 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip-only_vis
#python main.py --seed 228 --dataset ucfg1 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip-only_vis_seed228
#python main.py --seed 0 --dataset ucfg1 --feature-group vis --aggregate_text --extra_loss --use_dic_gt --feat-extractor clip --feature-size 512 --rgb-list ./list/ucfg1-clip.list --test-rgb-list ./list/ucfg1-clip-test.list --exp-name ucfg1-clip-only_vis_seed0

end_time=$(date +%s)
runtime_hours=$(((end_time - start_time)/3600))
echo "任务运行时间：$runtime_hours 小时"
date +"%Y-%m-%d %H:%M:%S"