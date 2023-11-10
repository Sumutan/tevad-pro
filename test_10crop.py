import matplotlib.pyplot as plt
import torch, sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import numpy as np
from utils import get_gt, anomap, compute_auc, compute_far
import pickle
import os

def pad_array(arr, length):  # padding 1-D ndarray by last element
    last_element = arr[-1]
    padding_length = length - len(arr)
    padding = [last_element] * padding_length
    if padding_length>0:
        return np.pad(arr, (0, padding_length), 'constant', constant_values=(0, last_element))
    else:
        return arr


def get_gt_dic(picklePath):
    with open(picklePath, 'rb') as f:
        frame_label = pickle.load(f)
    return frame_label


def test(dataloader, model, args, viz, device, plot_curve=False, logger=None, step=0):
    if args.use_dic_gt:
        if 'shanghai' in args.dataset:
            gt_dic = get_gt_dic('./list/gt-sh2-dic.pickle')
        elif 'ucf' in args.dataset:
            gt_dic = get_gt_dic('./list/gt-ucf-dic.pickle')
        elif 'TAD' in args.dataset:
            gt_dic = get_gt_dic('./list/gt-tad-dic.pickle')
        else:
            raise ValueError('Dataset not supported')

    snipit_length = 16 if args.sampling == 1 else 64

    with torch.no_grad():
        model.eval()
        pred, gt = torch.zeros(0), torch.zeros(0)
        pred_abn, gt_abn = torch.zeros(0), torch.zeros(0)  # 仅仅记录异常视频的预测结果与GT

        all_count, abn_count = 0, 0  # 正常/异常视频计数器
        predict_dict, gt_dict = {}, {}
        if args.use_dic_gt:
            for i, (input, text, fname) in enumerate(dataloader):  # test set has 199 videos
                input = input.to(device)
                input = input.permute(0, 2, 1, 3)
                text = text.to(device)
                text = text.permute(0, 2, 1, 3)
                # input.shape = (1,10,T,2048); T clips, each clip has 16(snipit_length)frames, each frame has 10 crops
                # https://github.com/tianyu0207/RTFM/issues/51
                # 使用时可以把10那一维拉平变成(1, 10*T, 2048), 中间那一维就是visual features再和caption进行concat操作
                score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
                    feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes \
                    = model(input, text)
                # 注意这里的score_abnormal和score_normal是一维的，是每一个video的一个分数，而logits则是一个T维的vector给每一个snippet都打了分

                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                sig = logits

                # 由于数据集长度不能被16整除，pred与gt不对齐，多切少补（gt）
                _gt_raw = gt_dic[fname[0]]

                if sig.shape[0] * 16 < len(_gt_raw):
                    _gt = torch.tensor(_gt_raw[:len(sig) * snipit_length])
                else:
                    _gt = torch.tensor(pad_array(_gt_raw, len(sig) * snipit_length))

                # if args.alignment_method == 'add':  # TEVAD default, len(pred*16)>=len(gt)
                #     # 由于pred的策略是对最后不足16(snipit_length)frame的视频以最后一个frame进行补齐，所以此处也对应的对gt进行补齐
                #     if sig.shape[0]*16<len(_gt_raw):
                #         _gt=torch.tensor(_gt_raw[:len(ig[0])*16])
                #     else:
                #         _gt = torch.tensor(pad_array(_gt_raw, len(sig) * snipit_length))
                # elif args.alignment_method == 'cut':  # 对于最后不足16(snipit_length)的frame直接丢弃不做推理，len(pred*16)<=len(gt)
                #     _gt = torch.tensor(_gt_raw[:len(sig) * snipit_length])
                # else:
                #     raise ValueError('Error: alignment method not specified')

                pred = torch.cat((pred, sig))  # pred means pread_all_preds
                gt = torch.cat((gt, _gt))

                all_count += 1
                if np.max(_gt_raw):  # 如果是异常视频（视频中有1p为1），拼接到异常列表中用于AUC_abn/far_abn计算
                    abn_count += 1
                    pred_abn = torch.cat((pred_abn, sig))
                    gt_abn = torch.cat((gt_abn, _gt))

                predict_dict[fname[0]] = np.repeat(np.squeeze(sig.cpu().numpy()), snipit_length)
                gt_dict[fname[0]] = _gt.cpu().numpy()

            gt = gt.cpu().numpy()
            gt_abn = gt_abn.cpu().numpy()
        else:  # TEVAD default implement
            for i, (input, text) in enumerate(dataloader):  # test set has 199 videos
                input = input.to(device)
                input = input.permute(0, 2, 1, 3)
                text = text.to(device)
                text = text.permute(0, 2, 1, 3)
                # input.shape = (1,10,T,2048); T clips, each clip has 16frames, each frame has 10 crops
                # https://github.com/tianyu0207/RTFM/issues/51
                # 使用时可以把10那一维拉平变成(1, 10*T, 2048), 中间那一维就是visual features再和caption进行concat操作
                score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
                    scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input,
                                                                                   text)  # 注意这里的score_abnormal和score_normal是一维的，是每一个video的一个分数，而logits则是一个T维的vector给每一个snippet都打了分
                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                sig = logits
                pred = torch.cat((pred, sig))

            gt = get_gt(args.dataset, args.gt)

        #tcc
        if step == 760:
            with open(os.path.join('pickle','best_result.pickle'),'wb') as file:
                pickle.dump(predict_dict,file)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), snipit_length)  # 数组中的每个元素重复16遍，即同一个clip中的16帧共享相同的预测结果

        pred_abn = list(pred_abn.cpu().detach().numpy())
        pred_abn = np.repeat(np.array(pred_abn), snipit_length)  # 数组中的每个元素重复16遍，即同一个clip中的16帧共享相同的预测结果

        ap = average_precision_score(list(gt), pred)
        print('ap : ' + str(ap))
        rec_auc_all, pr_auc_all, fpr_all, tpr_all = compute_auc(gt, pred, 'all')
        far_all = compute_far(gt, pred, 'all')
        if args.use_dic_gt:
            rec_auc_abn, pr_auc_abn, fpr_all, tpr_all = compute_auc(gt_abn, pred_abn, 'abnormal')
            far_abn = compute_far(gt_abn, pred_abn, 'abnormal')

        # plot_curve = True
        if plot_curve:  # 画图模式下只画图
            anomap(predict_dict, gt_dict, args.dataset + '_' + args.feat_extractor,
                   step, args.abn_curve_save_root)
            return

        # viz.plot_lines('pr_auc', pr_auc_all)
        # viz.plot_lines('auc', rec_auc_all)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr_all, fpr_all)

        viz.plot_lines('pr_auc', pr_auc_all)
        viz.plot_lines('rec_auc_all', rec_auc_all)
        viz.plot_lines('rec_auc_abn', rec_auc_abn)
        viz.plot_lines('far_all', far_all)
        viz.plot_lines('far_abn', far_abn)
        viz.lines('scores', pred)
        viz.lines('roc', tpr_all, fpr_all)

        if args.save_test_results:
            np.save('results/' + args.dataset + '_pred.npy', pred)
            np.save('results/' + args.dataset + '_fpr.npy', fpr_all)
            np.save('results/' + args.dataset + '_tpr.npy', tpr_all)
            # np.save('results/' + args.dataset + '_precision.npy', precision)
            # np.save('results/' + args.dataset + '_recall.npy', recall)
            np.save('results/' + args.dataset + '_auc.npy', rec_auc_all)
            np.save('results/' + args.dataset + '_ap.npy', ap)

        if logger:
            logger.log(
                f"Epoch {step}  rec_auc_all: {rec_auc_all:.4f}  rec_auc_abn: {rec_auc_abn:.4f}  far_all: {far_all:.4f} far_abn:{far_abn:.4f}")

        if args.use_dic_gt:
            return rec_auc_all, ap, rec_auc_abn, far_all, far_abn
        else:
            return rec_auc_all, ap, 0, far_all, 0
