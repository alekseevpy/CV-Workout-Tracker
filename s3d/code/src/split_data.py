import os
import glob
import shutil
import numpy as np
import random

from tqdm.auto import tqdm

seed = 42
np.random.seed(seed)
random.seed(seed)

TEST_SPLIT = 0.1
VALID_SPLIT = 0.1
TRAIN_SPLIT = 1.0 - VALID_SPLIT - TEST_SPLIT
TRAINING_PERCENTAGE = 1.0

ROOT_DIR = os.path.join('..', '..', 'full_workout_dataset','resized')
DST_ROOT = os.path.join('..', '..', 'full_workout_dataset', 'split')
os.makedirs(DST_ROOT, exist_ok=True)

all_classes = os.listdir(ROOT_DIR)

def copy_data(video_list, split='train'):
    for i, video_name in tqdm(enumerate(video_list), total=len(video_list)):
        class_name = video_name.split(os.path.sep)[-2]
        data_dir = os.path.join(DST_ROOT, split, class_name)
        file_name = video_name.split(os.path.sep)[-1]
        os.makedirs(os.path.join(data_dir), exist_ok=True)
        shutil.copy(
            os.path.join(video_name),
            os.path.join(data_dir, file_name)
        )

for class_name in all_classes:
    class_videos = glob.glob(os.path.join(ROOT_DIR, class_name, '*'))
    total_samples = len(class_videos)
    train_videos = class_videos[0:int(total_samples*TRAIN_SPLIT)]
    valid_videos = class_videos[
int(total_samples*TRAIN_SPLIT):int(total_samples*TRAIN_SPLIT+total_samples*VALID_SPLIT)
    ]
    test_videos = class_videos[
int(total_samples*VALID_SPLIT):int(total_samples*VALID_SPLIT+total_samples*TEST_SPLIT)
    ]
    
    copy_data(train_videos[:int(TRAINING_PERCENTAGE*len(train_videos))], 'train')
    copy_data(valid_videos, 'valid')
    copy_data(test_videos, 'test')
    