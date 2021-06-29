import os
from videos import *

text = []
vid = []

output_dir = 'op'
for dirs, subdirs, files in os.walk('videos/Validation_Set'):
    for file in files:

        if file.endswith('.txt'):
            text.append(os.path.join(dirs,file))
        elif file.endswith('.mp4'):
            vid.append(os.path.join(dirs,file))


for index_1,item in enumerate(text):
    for index_2, item in enumerate(vid):
        split_videos(text[index_1],vid[index_2],'videos_split/validation')
        print(f'{item} is done')