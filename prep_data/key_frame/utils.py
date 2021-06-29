import numpy as np
import time
import peakutils
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
    Key frame extraction using absolute difference between the pixel values of current frame and previous frame
'''



def plot_metrics(indices,  y_indices, FrameCount, absDiff, name):
    plt.plot(indices, y_indices, "x")
    l = plt.plot(FrameCount, absDiff, 'r-')
    plt.xlabel('Frames')
    plt.ylabel('Absoulute pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.legend(['Total number of key frames: {}'.format(len(indices))])
    name = name.strip().split('.')[0] + '.png'    #Change this specific to your directory
    plt.savefig(name, dpi=100)
    plt.show()


def keyframe(video_path, Thres, PlotMetrics=True):
    FrameCount = []
    absDiff = []
      
    
    Start_time = time.process_time()
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    while ret:
        ret, curr_frame = cap.read()

        if ret:
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
            diff = cv2.absdiff(curr_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)
            absDiff.append(non_zero_count)
            FrameCount.append(frame_no)
            prev_frame = curr_frame
        
    cap.release()
    
    y = np.array(absDiff)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)
    y_indices = y[indices]
    
    
    
    
    if(PlotMetrics):
        plot_metrics(indices, y_indices, FrameCount, absDiff, video_path)

    data = {'name':video_path.strip().split('/')[2], 'frame':indices}  #change according to your directory
    df = pd.DataFrame(data)


    return df