import torch.utils.data as data
import numpy as np
from utils import process_feat, get_rgb_list_file
import torch
from torch.utils.data import DataLoader
import re
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        # self.modality = args.modality
        self.emb_folder = args.emb_folder
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.feature_size = args.feature_size
        self.use_dic_gt = args.use_dic_gt
        if args.test_rgb_list is None:
            _, self.rgb_list_file = get_rgb_list_file(args.dataset, test_mode, args.feat_extractor)
        else:
            if test_mode is False:
                self.rgb_list_file = args.rgb_list
            else:
                self.rgb_list_file = args.test_rgb_list

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.feat_extractor = args.feat_extractor
        self.caption_extractor = args.caption_extractor

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:  # list for training would need to be ordered from normal to abnormal
            if 'shanghai' in self.dataset:
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
            elif 'ucfg2' in self.dataset:
                if self.is_normal:
                    self.list = self.list[926:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:926]
                    print('abnormal list for ucf')
            elif 'ucfg1' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1107:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:1107]
                    print('abnormal list for ucf')
            elif 'ucf25%' in self.dataset:
                if self.is_normal:
                    self.list = self.list[195:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:195]
                    print('abnormal list for ucf')
            elif 'ucf50%' in self.dataset:
                if self.is_normal:
                    self.list = self.list[433:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:433]
                    print('abnormal list for ucf')
            elif 'ucf75%' in self.dataset:
                if self.is_normal:
                    self.list = self.list[615:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:615]
                    print('abnormal list for ucf')
            elif 'ucf' in self.dataset:
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
            elif 'TAD' in self.dataset:
                if self.is_normal:
                    self.list = self.list[190:]
                    print('normal list for TAD')
                else:
                    self.list = self.list[:190]
                    print('abnormal list for TAD')
            elif 'violence' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1904:]
                    print('normal list for violence')
                else:
                    self.list = self.list[:1904]
                    print('abnormal list for violence')
            elif 'ped2' in self.dataset:
                if self.is_normal:
                    self.list = self.list[6:]
                    print('normal list for ped2', len(self.list))
                else:
                    self.list = self.list[:6]
                    print('abnormal list for ped2', len(self.list))
            elif 'TE2' in self.dataset:  # 注意index从0开始，而pycharm行号从1开始
                if self.is_normal:
                    self.list = self.list[23:]
                    print('normal list for TE2', len(self.list))
                else:
                    self.list = self.list[:23]
                    print('abnormal list for TE2', len(self.list))
            else:
                raise Exception("Dataset undefined!!!")

    def _get_text_emb_path(self,vis_feature_path):
        postfix_length_dic={'videomae':len('videomae.npy'), #12
                        'clip':len('clip.npy'),  # 8
                        'i3d':len('i3d.npy')}    # 7
        postfix_length=postfix_length_dic[self.feat_extractor]
        if 'violence' in self.dataset and 'clip' in self.feat_extractor: postfix_length=4
        if 'ucf' in self.dataset:
            text_path = "save/Crime/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "emb.npy"
        elif 'shanghai' in self.dataset:
            text_path = "save/Shanghai/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "emb.npy"
        elif 'violence' in self.dataset:
            text_path = "save/Violence/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "_emb.npy"
        elif 'ped2' in self.dataset:
            text_path = "save/UCSDped2/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "emb.npy"
        elif 'TE2' in self.dataset:
            text_path = "save/TE2/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "emb.npy"
        elif 'TAD' in self.dataset:
            text_path = "save/TAD/" + self.emb_folder + "/" + vis_feature_path.split("/")[-1][:-postfix_length] + "emb.npy"
        else:
            raise Exception("Dataset undefined!!!")

        return text_path
    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1
        vis_feature_path = self.list[index].strip('\n')

        features = np.load(vis_feature_path, allow_pickle=True)  # allow_pickle允许读取其中的python对象
        features = np.array(features, dtype=np.float32)

        if features.shape[0] in [10,5]: # 10 crop
            features = features.transpose(1, 0, 2)  # [10,no.,768]->[no.,10,768]

        text_path=self._get_text_emb_path(vis_feature_path)
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]
        # assert features.shape[0] == text_features.shape[0]

        if self.caption_extractor == 'swinBERT':
            if 'violence' in self.dataset:
                text_features = np.tile(text_features, (10, 1, 1))  # [5,snippet no.,768]
            elif self.feature_size in [2560,2048, 1536, 1280 ,1024, 768, 512]:  # vis feature 是按10_crop提的，但text提1crop，所以tile对齐维度
                text_features = np.tile(text_features, (10, 1, 1))  # [10,snippet no.,768]
            else:
                raise Exception("Feature size undefined!!!")

        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)  # [snippet no.,10,768]

            if self.use_dic_gt:
                if self.feat_extractor == 'videomae':
                    lableFileName = self.list[index].split('/')[-1].split('_videomae')[0]
                elif self.feat_extractor == 'i3d':
                    lableFileName = self.list[index].split('/')[-1].split('_i3d')[0]
                elif self.feat_extractor == 'clip':
                    lableFileName = self.list[index].split('/')[-1].split('_clip')[0]
                else:
                    raise NotImplementedError
                return features, text_features, lableFileName.split('.npy')[0] #return fileName

            return features, text_features  # 原代码的输出
        else:  # train mode
            # process 10-cropped snippet feature
            if features.shape[1] in [10, 5]:  # 10 crop
                features = features.transpose(1, 0, 2)  # [snippet no., 10, 2048] -> [10, snippet no., 2048]
            divided_features = []
            for feature in features:  # loop 10 times  [10,snippet no.,2048]->[10,32,2048]
                feature = process_feat(feature, 32)  # divide a video into 32 segments/snippets/clips
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)  # [10,32,2048]

            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)  # [32,768]
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)
            assert divided_features.shape[1] == div_feat_text.shape[1], str(self.test_mode) + "\t" + str(
                divided_features.shape[1]) + "\t" + div_feat_text.shape[1]
            return divided_features, div_feat_text, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


class Dataset_DualVisionEncoder(Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        super().__init__(args, is_normal, transform, test_mode)
        self.feat_extractor_B = args.feat_extractor_B # clip
        self.feature_size_B = args.feature_size_B
        self.vis_feature_path_B_folder = args.vis_feature_path_B_folder
        print()

    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1
        vis_feature_path = self.list[index].strip('\n')
        vis_feature_path_B = os.path.join(self.vis_feature_path_B_folder,vis_feature_path.split('/')[-1]).replace(self.feat_extractor,self.feat_extractor_B)

        #load feature A
        features = np.load(vis_feature_path, allow_pickle=True)  # allow_pickle允许读取其中的python对象
        features = np.array(features, dtype=np.float32)
        if features.shape[0] in [10, 5]:  # 10 crop
            features = features.transpose(1, 0, 2)  # [10,no.,768]->[no.,10,768]
        #load feature B
        features_B = np.load(vis_feature_path_B, allow_pickle=True)  # allow_pickle允许读取其中的python对象
        features_B = np.array(features_B, dtype=np.float32)
        if features_B.shape[0] in [10, 5]:  # 10 crop
            features_B = features_B.transpose(1, 0, 2)  # [10,no.,768]->[no.,10,768]
            if features_B.shape[0]<features.shape[0]:  # 如果应为提取特征时对最后不足16frames的帧的保留策略不同导致多1段，则补上
                lastframe=features_B[-1,:,:]
                lastframe = np.expand_dims(lastframe, axis=0)
                features_B = np.concatenate((features_B, lastframe), axis=0)
            if features_B.shape[0]>features.shape[0]:  # 如果应为提取特征时对最后不足16frames的帧的保留策略不同导致多1段，则补上
                lastframe=features[-1,:,:]
                lastframe = np.expand_dims(lastframe, axis=0)
                features = np.concatenate((features, lastframe), axis=0)
        #concat visual feature
        features=np.concatenate((features,features_B),axis=-1)

        text_path = self._get_text_emb_path(vis_feature_path)
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]
        # assert features.shape[0] == text_features.shape[0]

        if self.caption_extractor == 'swinBERT':
            if 'violence' in self.dataset:
                text_features = np.tile(text_features, (5, 1, 1))  # [5,snippet no.,768]
            elif self.feature_size in [2560, 2048, 1536, 1280, 1024, 768,
                                       512]:  # vis feature 是按10_crop提的，但text提1crop，所以tile对齐维度
                text_features = np.tile(text_features, (10, 1, 1))  # [10,snippet no.,768]
            else:
                raise Exception("Feature size undefined!!!")

        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)  # [snippet no.,10,768]

            if self.use_dic_gt:
                if self.feat_extractor == 'videomae':
                    lableFileName = self.list[index].split('/')[-1].split('_videomae')[0]
                elif self.feat_extractor == 'i3d':
                    lableFileName = self.list[index].split('/')[-1].split('_i3d')[0]
                elif self.feat_extractor == 'clip':
                    lableFileName = self.list[index].split('/')[-1].split('_clip')[0]
                else:
                    raise NotImplementedError
                return features, text_features, lableFileName.split('.npy')[0]  # return fileName

            return features, text_features  # 原代码的输出
        else:  # train mode
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [snippet no., 10, 2048] -> [10, snippet no., 2048]
            divided_features = []
            for feature in features:  # loop 10 times  [10,snippet no.,2048]->[10,32,2048]
                feature = process_feat(feature, 32)  # divide a video into 32 segments/snippets/clips
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)  # [10,32,2048]

            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)  # [32,768]
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)
            assert divided_features.shape[1] == div_feat_text.shape[1], str(self.test_mode) + "\t" + str(
                divided_features.shape[1]) + "\t" + div_feat_text.shape[1]
            return divided_features, div_feat_text, label
