import numpy as np
import os

file1 = np.load("Normal_Videos307_x264_videomae.npy")
file2 = np.load("concated_Normal_Videos307_x264.npy")
pass

# file1 = np.load(os.path.join(dir_mae, f))
# file2 = np.load(os.path.join(dir_mae2, f))
# if file1.shape[:2]!= file2.shape[:2]:
#     print(f"{f}:{file1.shape} != {file2.shape}")