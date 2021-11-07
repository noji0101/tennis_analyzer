import os
import sys
import numpy as np

sys.path.append(os.path.join('modules/simple_HRNet'))
sys.path.append(os.path.join('modules/swing_classifier'))
sys.path.append(os.path.join('modules/TrackNet'))
sys.path.append(os.path.join('modules/simple_HRNet/models/detectors'))

from utils.load import load_yaml
from utils.make_video import make_video
from modules.simple_HRNet.scripts.extract_keypoints import extract_keypoints
from modules.simple_HRNet.scripts.make_video import make_video
from modules.swing_classifier.executor.inferer import SwingInferer
from modules.TrackNet.predict_video_coordinates import Ball_detector
from modules.analyzer.make_dataframe import make_dataframe

def main(movie_path, configfile, court_points, is_output_movie=False): # , court_points

    config = load_yaml(configfile)

    print('\n==================== pose estimation ====================')

    points = extract_keypoints(movie_path, config['HRnet'])
    np.savetxt('app/static/uploads/test.csv', points)


    print('\n==================== swing classifitation ====================')

    # points = np.loadtxt('data/input/my_tennis_1/output.csv', delimiter=',')
    swing_inferer = SwingInferer(points, config['Swing_clssifier'])
    swing_classes = swing_inferer.predict()

    print('\n==================== ball detection ====================')

    ball_detector = Ball_detector(movie_path, config['TrackNet'])
    ball_coordinates = ball_detector.predict_ball_coordinates()

    print('\n==================== Analyze ====================')

    all_data = make_dataframe(points, swing_classes, ball_coordinates, court_points)
    return all_data


if __name__=='__main__':
    court_points = {'court_point_1': [1130, 594],
                'court_point_2': [782, 477],
                'court_point_3': [397, 461],
                'court_point_4': [3, 578]}
    main('my_tennis_20sec.mp4', 'config/config.yaml', court_points) 


