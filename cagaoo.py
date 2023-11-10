import numpy as np
import os

# file1 = np.load("01_Accident_001_i3d.npy")
# file2 = np.load("01_Accident_001_videomae.npy")
# pass
dir_i3d = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_i3d"
dir_mae = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_11-1_10-15_finetune_L"
dir_mae2 = "/media/sh/9a898338-9715-47b4-bca4-43f2270c463a/sh/programs/TEVAD-main/save/TAD/TAD_9-5_9-1_finetune_dif_0.5_SP_norm"

# for f in os.listdir(dir_i3d):
f="01_Accident_002_i3d.npy"
file1 = np.load(os.path.join(dir_i3d, f))
file2 = np.load(os.path.join(dir_mae2, f.replace('i3d', 'videomae')))
if file1.shape[:2]!= file2.shape[:2]:
    print(f"{f}:{file1.shape} != {file2.shape}")

# file1 = np.load(os.path.join(dir_mae, f))
# file2 = np.load(os.path.join(dir_mae2, f))
# if file1.shape[:2]!= file2.shape[:2]:
#     print(f"{f}:{file1.shape} != {file2.shape}")