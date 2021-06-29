import cv2
import os
def get_frame_numbers(txt_file):
    with open(txt_file, 'r') as file:
        next(file)
        frame_numbers = file.read().split()

    return frame_numbers

def get_frames(vid):
    while True:
        ret, frame = vid.read()
        if not ret:
            return
        yield frame

def split_videos(txt_file,video_file,output_dir):
    new_videos = {}
    video = cv2.VideoCapture(video_file)
    codec = cv2.VideoWriter_fourcc(*'XVID')

    frame_numbers = get_frame_numbers(txt_file)
    count = 0
    for index, frame in zip(frame_numbers,get_frames(video)):
        if index not in new_videos:
            count+=1
            filename = video_file.replace('/',' ').replace('.',' ').replace('/', ' ').split()[2]
            filepath = f'{output_dir}/{index}'  
            if os.path.isdir(filepath) == False:
                os.makedirs(filepath)
              
            savepath = f'{output_dir}/{index}/{filename}_{count}.avi'
          
            new_videos[index] = cv2.VideoWriter(savepath, codec, 30.0, (len(frame[0]), len(frame)))
        new_videos[index].write(frame)
    video.release()
    for vid in new_videos.values():
        vid.release()
    cv2.destroyAllWindows()


