import os
import sys
import argparse
import ast
import csv
import cv2
import time
import torch
import numpy as np



sys.path.append('/home/duser/workspace/tennis_analyzer/modules/simple_HRNet/scripts/../')
sys.path.append('/home/duser/workspace/tennis_analyzer/modules/simple_HRNet/models/detectors')
sys.path.append('/home/duser/workspace/tennis_analyzer/')
sys.path.append('/home/duser/workspace/tennis_analyzer/modules')
sys.path.insert(1, 'modules/simple_HRNet')

from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation


def extract_keypoints(filename, config):
    
    hrnet_m = config['hrnet_m']
    hrnet_j = config['hrnet_j']
    hrnet_c = config['hrnet_c']
    hrnet_weights = config['hrnet_weights']
    image_resolution = config['image_resolution']
    single_person  = config['single_person']
    max_batch_size = config['max_batch_size']
    save_csv = config['save_csv']
    csv_output_filename = config['csv_output_filename']
    device = config['device']
            
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    image_resolution = ast.literal_eval(image_resolution)

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if save_csv:
        assert csv_output_filename.endswith('.csv')
        os.makedirs(f'./result/{filename[0:-4]}', exist_ok=True)
        fd = open(f'result/{filename[0:-4]}/{csv_output_filename}', 'wt', newline='')
        csv_output = csv.writer(fd, delimiter=',')

        print(f'result/{filename[0:-4]}/{csv_output_filename}')
    
    yolo_model_def = "modules/simple_HRNet/models/detectors/yolo/config/yolov3.cfg"
    yolo_class_path = "modules/simple_HRNet/models/detectors/yolo/data/coco.names"
    yolo_weights_path = "modules/simple_HRNet/models/detectors/yolo/weights/yolov3.weights"

    if __name__ == '__main__':
        yolo_model_def = "models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path = "models/detectors/yolo/data/coco.names"
        yolo_weights_path = "models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    points = []
    index = 0
    while True:
        t = time.time()

        ret, frame = video.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        pts = model.predict(frame)

        # csv format is:
        #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
        # where each <point N> corresponds to three elements:
        #   y_coordinate,x_coordinate,confidence

        for j, pt in enumerate(pts):
            row = [index, j] + pt.flatten().tolist()
            points.append(row)
            if save_csv:
                csv_output.writerow(row)

        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1

    if save_csv:
        fd.close()
    else:
        return np.array(points)


if __name__ == '__main__':
    config = {'hrnet_m': 'HRNet',
    'hrnet_c': 48,
    'hrnet_j': 17,
    'hrnet_weights': '/home/duser/workspace/tennis_analyzer/data/simple_HRnet/weights/pose_hrnet_w48_384x288.pth',
    'image_resolution': '(384, 288)',
    'single_person': False,
    'max_batch_size': 16,
    'save_csv': True,
    'csv_output_filename': 'tendency_validation.csv',
    'device': None}
    extract_keypoints('tendency_validation.mp4', config)