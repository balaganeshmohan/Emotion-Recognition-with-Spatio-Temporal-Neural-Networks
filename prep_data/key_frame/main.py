import numpy as np
import time
import peakutils
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
from utils import *

'''
    This is written to parse through the videos in my directories. Edit the main function according to your need. The function takes
    directories as input and returns a csv file with the list of key frames per video. 

'''

parser = argparse.ArgumentParser(description='KeyFrameExtraction')
parser.add_argument('--source_dir',default=None, help='Enter video directory')
parser.add_argument('--Thres',default=None, type=float)
parser.add_argument('--plot_metrics',default=True)

args = parser.parse_args()

def main():
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()

    excludes = ['Test_Set']
    for dirs,dirnames, files in os.walk('videos'):
        if "Test_Set" in dirnames:
            dirnames.remove("Test_Set")
        for file in files:
            if file.endswith('.mp4'):
                path = os.path.join(dirs,file)
                df = keyframe(args.source_dir, args.Thres)
                if path.strip().split('/')[1] ==  "Train_Set":
                    df_train = df_train.append(df, ignore_index=True)
                else:
                    df_val = df_val.append(df, ignore_index=True)

    df_train.to_csv('train.csv', header=None, sep =" ")
    df_val.to_csv('val.csv', header=None, sep=" ")


if __name__ == '__main__':
    main()


