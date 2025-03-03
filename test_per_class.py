from eval import eval_p
from torch.utils.data import Dataset, DataLoader
import time
import torch
import os
import argparse
import numpy as np
import pickle
import json

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # save_path = 'curve'
    pickle_file='results/ucf/UCF-videoMAE-videoMaeST-FBseparation_threthold10-CLIP-B/_pred.pickle'
    with open(pickle_file, 'rb') as file:
        result = pickle.load(file)
    test_list = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism',] #UCF
    # test_list = ['01_Accident','02_IllegalTurn','03_IllegalOccupation','04_Retrograde','05_else','06_PedestrianOnRoad','07_RoadSpills',]  #TAD

    #draw plots
    for j in range(len(test_list)):
        name = test_list[j]

        result_new = {}
        for key in result.keys():
            if name in key:
                result_new[key] = result[key]
        print('AUC of ',name,':')
        eval_p(predict_dict=result_new, plot=False,dataset='ucf') #dataset='ucf'or'TAD'


