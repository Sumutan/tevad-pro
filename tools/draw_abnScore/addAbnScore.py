# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pickle
import os


def readPickle(picklePath):
    assert os.path.exists(picklePath), f"pickle file {picklePath} is not exist."
    # 打开pickle文件
    with open(picklePath, 'rb') as f:
        data = pickle.load(f)
    return data


def video_to_numpy_array(video_path):
    """
        input:
            video_path:视频路径
        output:
            video_nparray:[t,h,w,3] RGB
    """
    rgb_frame_list = []
    video_read_capture = cv2.VideoCapture(video_path)
    while video_read_capture.isOpened():
        result, frame = video_read_capture.read()
        if not result:
            break

        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将帧转换为RGB格式
        # rgb_frame_list.append(rgb_frame)
        rgb_frame_list.append(frame)

    video_read_capture.release()

    video_nparray = np.array(rgb_frame_list)

    return video_nparray  # video_nparray:[t,h,w,3] RGB


def numpy_array_to_video(numpy_array, video_out_path, videoType='mp4', fps=25):
    # 获取视频的高度和宽度
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]
    out_video_size = (video_width, video_height)

    # 定义输出视频的fourcc编解码器，用于写入视频帧
    if videoType.lower() == 'mp4':
        output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    elif videoType.lower() == 'avi':
        output_video_fourcc = int(cv2.VideoWriter_fourcc(*'avi'))
    else:
        raise ValueError

    # 创建一个VideoWriter对象，用于将输出视频帧写入到指定路径
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, fps, out_video_size)

    # 循环遍历numpy数组中的每一帧，并将其写入输出视频
    for frame in numpy_array:
        video_write_capture.write(frame)

    # 释放VideoWriter对象
    video_write_capture.release()

def add_Border_and_AbnScore_cv(image, abn=0.00):
    h, w, c = image.shape

    if abn > 0.5:
        a, b = (0, 0), (w, h)
        cv2.rectangle(image, a, b, (0, 0, 255), 5)

    # 在右上角添加文本 "abn"
    text = f"abnScore:{abn:.2f}"
    cv2.putText(image, text, (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #                           位置               字体                 大小   颜色       字体厚度

def addAbnScore(video_input_path, video_output_path):
    video_nparray = video_to_numpy_array(video_input_path)  # video_array:[t,h,w,3] RGB
    gt_dic = readPickle('predict_dict.pkl')
    video_basename = os.path.basename(video_input_path).split('.')[0]
    gt = gt_dic[video_basename]

    gt = gt[:video_nparray.shape[0]]

    for i, frame in enumerate(video_nparray):
        # add_Border_and_AbnScore(frame, gt[i])
        add_Border_and_AbnScore_cv(frame, gt[i])
    numpy_array_to_video(video_nparray, video_output_path, videoType='mp4', fps=25)
    print(video_output_path)


if __name__ == '__main__':
    #获取待处理文件列表
    folder_path = 'tmp'  # 替换为实际文件夹的路径
    video_input_path_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.avi')]     # 获取文件夹中所有以 .avi 格式结尾的文件路径
    # video_input_path_list = [r'01_0015.avi']  #单个文件测试
    video_output_path_list = [filename.split('.')[0] + '.mp4' for filename in video_input_path_list]

    for video_input_path,video_output_path in zip(video_input_path_list,video_output_path_list):
        addAbnScore(video_input_path, video_output_path)
