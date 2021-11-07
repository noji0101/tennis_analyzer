import numpy as np
import torch
import matplotlib.pyplot as plt


def judge_sequence(num_list):
    '''Check for missing frame numbers'''
    leakages = []
    total_diff = 0
    for i in range(len(num_list)):
        if (i + total_diff) != num_list[i]:
            diff = num_list[i] - (i + total_diff)
            for j in range(diff):
                leakages.append(i + total_diff + j)
            total_diff += diff
    return leakages

def preprocess_points(points, delete_weight=2.1, is_for_analysis=False):

    # Remove data that is expected to be non-player
    # calculate person size
    person_sizes = []
    for i, point in enumerate(points):
        left_eye = np.array([point[5], point[6]])
        right_eye = np.array([point[8], point[9]])
        left_ankle = np.array([point[47], point[48]])
        right_ankle = np.array([point[50], point[51]])
        
        distance_1 = np.linalg.norm(left_eye - right_ankle)
        distance_2 = np.linalg.norm(right_eye - left_ankle)
        
        person_sizes.append(np.amax(np.array(distance_1, distance_2)))
    points= np.append(points, np.array([[i] for i in person_sizes]), axis=1)

    #　delete all but the largets person each frame
    all_idx_delete = np.array([])
    for frame in range(np.int(points[-1, 0])): # Set the frame of the last line as the total number of frames
        indices = np.where(points[:, 0] == frame)
        try:
            idx_max = np.argmax(points[indices, -1])
            idx_delete = np.delete(indices, idx_max)
            all_idx_delete = np.append(all_idx_delete, idx_delete)
        except ValueError:
            pass
    all_idx_delete = all_idx_delete.astype(np.int64)
    points = np.delete(points, all_idx_delete, 0)

    # Delete outliers
    # Calculate the mean and variance of all frames of person_sizes
    leargets_person_sizes = points[:, -1]
    delete_threshold = leargets_person_sizes.mean() -  delete_weight * leargets_person_sizes.std()
    idx_delete = np.where(leargets_person_sizes > delete_threshold)
    points = points[idx_delete]

    # Exclude confidence and person's ID
    points = points[:, 
                    [0, # frame 
                    2, 3, # nose
                    5, 6, # left_eye
                    8, 9, # ...
                    11, 12,
                    14, 15,
                    17, 18,
                    20, 21,
                    23, 24,
                    26, 27,
                    29, 30,
                    32, 33,
                    35, 36,
                    38, 39,
                    41, 42,
                    44, 45,
                    47, 48,
                    50, 51
                    ]
                    ]

    # Convert frame number to int
    for idx in range(len(points)):
        points[idx, 0] = np.int(points[idx, 0])
    
    if not is_for_analysis:
        # Move the origin of the coordinates to the center of the shoulder
        for frame in range(len(points)):
            x_new_origin = (points[frame, 11] + points[frame, 13]) / 2
            y_new_origin = (points[frame, 12] + points[frame, 14]) / 2
            for i in range(1, 2*17+1):
                if i % 2 == 1:
                    points[frame, i] -= x_new_origin
                else:
                    points[frame, i] -= y_new_origin

        # Check for missing frame numbers
        leakages = judge_sequence([np.int(i) for i in points[:, 0]])

        # Add the missing frame number and set the coordinates to 999.
        for idx in leakages:
            points = np.insert(points, idx, np.insert(np.full(34, 999), 0, idx), axis=0)
        # delete frame num
        points = points[:, 1:]

        points = torch.from_numpy(points).float()
        # Transform the shape of the data to adapt to LSTM
        points = points.view(len(points), 1, -1)

    return points

def preprocess_anno(targets, points):
    ############## anno ###############
    targets = targets.astype(np.int)
    # フレーム数をkeypointsに合わせる
    targets = targets[:len(points)]
    
    # 関節情報がないフレームは強制的に測定不可能のクラスに変更
    targets[np.where(points[:, 0, 0] == 999)] = 8

    targets = torch.from_numpy(targets).clone()

    return targets

