import os
import cv2

import pandas as pd


def is_alowed_file(filename, allowed_extension):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

def get_total_frame_num(video):
    count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return count

def save_frame(video, frame_num, result_path):

    if not video.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = video.read()

    if ret:
        cv2.imwrite(result_path, frame)

def list_to_dataframe(lst):
    columns = ['frame', 'swing_classes', 'x_player', 'y_player', 'x_player_converted',
       'y_player_converted', 'x_ball', 'y_ball', 'x_ball_converted',
       'y_ball_converted']
    dataframe = pd.DataFrame(lst, columns=columns)
    dataframe['frame'] = dataframe['frame'].astype('int')
    dataframe['swing_classes'] = dataframe['swing_classes'].astype('int')

    return dataframe
    