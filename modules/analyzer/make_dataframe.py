import pandas as pd
import numpy as np
import cv2

from modules.swing_classifier.dataloader.preprocess import preprocess_points


def make_dataframe(points, swing_classes, ball_coordinates, court_points):
    # pointsから人の足の座標
    player_coordinates = get_player_coordinates(points)

    # read court points
    court_points = np.array([court_points['court_point_1'], court_points['court_point_2'], 
                court_points['court_point_3'], court_points['court_point_4']])

    # 台形補正の行列獲得
    homography_matrix = get_homography_matrix(court_points)

    modified_swing_classes = modify_swing_class(swing_classes)

    # dataframeに統合
    df_swingclass = pd.DataFrame({ 'frame' : range(len(swing_classes)),
                                'swing_classes' : swing_classes,
                                'modified_swing_classes' : modified_swing_classes})
    player_coordinates_converted = player_coordinates.copy()
    player_coordinates_converted[:, 1:] = cv2.perspectiveTransform(np.array([player_coordinates_converted[:, 1:]]), homography_matrix)

    df_player_coordinates = pd.DataFrame({ 'frame' : player_coordinates.astype(np.int32)[:, 0],
                                            'x_player' : player_coordinates[:, 1],
                                            'y_player' : player_coordinates[:, 2], 
                                            'x_player_converted' : player_coordinates_converted[:, 1],
                                            'y_player_converted' : player_coordinates_converted[:, 2], })

    ball_coordinates_converted = ball_coordinates.copy().astype(np.float32)
    ball_coordinates_converted[:, 1:] = cv2.perspectiveTransform(np.array([ball_coordinates_converted[:, 1:]]), homography_matrix)

    df_ball_coordinates = pd.DataFrame({ 'frame' : ball_coordinates[:, 0],
                                            'x_ball' : ball_coordinates[:, 1],
                                            'y_ball' : ball_coordinates[:, 2],
                                            'x_ball_converted' : ball_coordinates_converted[:, 1],
                                            'y_ball_converted' : ball_coordinates_converted[:, 2]})                                   
    df_all = pd.merge(df_swingclass, df_player_coordinates, on='frame', how='left')
    df_all = pd.merge(df_all, df_ball_coordinates, on='frame', how='left')

    # y座標が極端にコートから離れている行は、プレイヤーの座標をNaNに
    delete_indices = df_all[(df_all['y_player_converted'] < 0) | (df_all['y_player_converted'] > 2000)].index.tolist()
    df_all.loc[delete_indices, ['x_player', 'y_player', 'x_player_converted', 'y_player_converted']] = np.nan
    return df_all

def get_player_coordinates(points):
    flatten_points = preprocess_points(points, is_for_analysis=True)
    points = np.empty([len(flatten_points), 17, 2])
    for i, point in enumerate(flatten_points):
        joint_points = np.empty([17, 2])
        joint_points[0] = [point[0], None]
        for j in range(1,17):
            joint_points[j] = np.array([point[2*j + 2], point[2*j+1]])
        points[i] = joint_points
    
    player_x_y = np.array([(points[:, 15, 0] + points[:, 16, 0]) / 2, (points[:, 15, 1] + points[:, 16, 1]) / 2 - (points[:, 5, 1] - points[:, 11, 1])/3]).transpose()
    player_coordinates = np.empty([len(points), 3], dtype=np.float32)
    for i, frame in enumerate(points[:, 0]):
        player_coordinates[i] = np.insert(player_x_y[i], 0, np.int(frame[0]))

    return player_coordinates

def get_homography_matrix(court_points):
    court_image_points = np.array([court_points], dtype=np.float32)
    court_actual_points = np.array([[823, 1190], [823, 0], [0, 0], [0, 1190]], dtype=np.float32) # actual tennis court size
    homography_matrix = cv2.getPerspectiveTransform(court_image_points, court_actual_points)
    print(cv2.perspectiveTransform(np.array(court_image_points), homography_matrix))
    
    return homography_matrix

def count_sequence(array):
    change = (array[1:] == array[:-1])

    left = np.arange(len(change))
    left[change] = 0
    np.maximum.accumulate(left, out = left)

    right = np.arange(len(change))
    right[change[::-1]] = 0
    np.maximum.accumulate(right, out = right)
    right = len(change) - right[::-1] -1

    count_list = np.zeros_like(array)
    count_list[:-1] += right
    count_list[1:] -= left
    count_list[-1] = 0

    return count_list

def modify_swing_class(swing_classes, threshold=5):
    sequence_count_list = count_sequence(swing_classes)
    noise_indices = np.where(sequence_count_list < threshold)
    modified_swing_classes = swing_classes.copy()
    modified_swing_classes[noise_indices] = 0

    return modified_swing_classes
