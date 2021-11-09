import math

import numpy as np
import pandas as pd
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature

def tendancy(all_data, court_points, fps, video_width):
    '''
    This function predict 
    0:other
    1:center
    2:left
    3:right
    '''
    region_points = calc_region_points(court_points, video_width)
    polygons = make_polygons(region_points)

    forehand_frames, backhand_frames = detect_fore_back_frame(all_data)

    forehand_regions = get_region_classes(all_data, forehand_frames, fps, polygons)
    backhand_regions = get_region_classes(all_data, backhand_frames, fps, polygons)
    
    
    region_ratio_forehand = get_region_ratio(forehand_regions, player_position=None, all_data=all_data)
    region_ratio_backhand = get_region_ratio(backhand_regions, player_position=None, all_data=all_data)
    region_ratio_left_forehand = get_region_ratio(forehand_regions, player_position='left', all_data=all_data)
    region_ratio_right_forehand = get_region_ratio(forehand_regions, player_position='right', all_data=all_data)
    region_ratio_left_backhand = get_region_ratio(backhand_regions, player_position='left', all_data=all_data)
    region_ratio_right_backhand = get_region_ratio(backhand_regions, player_position='right', all_data=all_data)


    return [region_ratio_forehand, region_ratio_backhand, region_ratio_left_forehand, region_ratio_right_forehand, region_ratio_left_backhand, region_ratio_right_backhand]


def calc_region_points(court_points, video_width):
    bottom_right = np.array(court_points['court_point_1'])
    top_right = np.array(court_points['court_point_2'])
    top_left = np.array(court_points['court_point_3'])
    bottom_left = np.array(court_points['court_point_4'])


    top_center_left = (1.1*top_right + 2*top_left)/3.1
    top_center_right = (2*top_right + 1.1*top_left)/3.1
    bottom_center_left = (3*bottom_right + 4*bottom_left)/7
    bottom_center_right = (4*bottom_right + 3*bottom_left)/7
    middle_center_left = (1*bottom_left + 2*top_left)/3
    middle_center_right = (1*bottom_right + 2*top_right)/3

    # 
    intersection_1 = calc_intersection(top_center_left, bottom_center_left, top_center_right, bottom_center_right)
    intersection_2 = calc_intersection(top_center_left, bottom_center_left, middle_center_left, middle_center_right)
    intersection_3 = calc_intersection(top_center_right, bottom_center_right, middle_center_left, middle_center_right)
    intersection_4 = calc_intersection(intersection_2, intersection_3, [0, 0], [0, 1])
    intersection_5 = calc_intersection(intersection_2, intersection_3, [video_width-1, 0], [video_width-1, 1])
    delta = bottom_center_left - bottom_center_right
    temporary_point = intersection_1 + delta
    intersection_6 = calc_intersection(intersection_1, temporary_point, [0, 0], [0, 1])
    intersection_7 = calc_intersection(intersection_1, temporary_point, [video_width-1, 0], [video_width-1, 1])

    region_points = [intersection_1, intersection_2, intersection_3, intersection_4, intersection_5, intersection_6, intersection_7] 

    return region_points

def calc_intersection(point1, point2, point3, point4):
    x1, y1, x2, y2, x3, y3, x4, y4 = point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]
    if x1 == x2 and x3 == x4:
        x = y = np.nan
    elif x1 == x2:
        x = x1
        y = (y4 - y3) / (x4 - x3) * (x1 - x3) + y3
    elif x3 == x4:
        x = x3
        y = (y2 - y1) / (x2 - x1) * (x3 - x1) + y1
    else:
        a1 = (y2-y1)/(x2-x1)
        a3 = (y4-y3)/(x4-x3)
        if a1 == a3:
            x = y = np.nan
        else:
            x = (a1*x1-y1-a3*x3+y3)/(a1-a3)
            y = (y2-y1)/(x2-x1)*(x-x1)+y1
    return (x, y)

def make_polygons(intersections):
    center_polygon = Polygon([[
                intersections[0],
                intersections[1],
                intersections[2]
                ]])
    left_polygon = Polygon([[
                intersections[0],
                intersections[1],
                intersections[3],
                intersections[5]
                ]])
    right_polygon = Polygon([[
                intersections[0],
                intersections[2],
                intersections[4],
                intersections[6]
                ]])
    polygons = [center_polygon, left_polygon, right_polygon]

    return polygons

def judge_region(points, polygons):
    '''
    0:other
    1:center
    2:left
    3:right
    '''
    region_classes = []
    geopoints = [Feature(geometry=Point(point)) for point in points]

    for index, geopoint in enumerate(geopoints):
        for i, polygon in enumerate(polygons):
            if boolean_point_in_polygon(geopoint, polygon):
                resion_class = i + 1
                region_classes.append(resion_class)
                
        if len(region_classes) != (index + 1):
            region_classes.append(0)
    return region_classes

def detect_fore_back_frame(all_data):
    forehand_frames = []
    backhand_frames = []
    for frame in range(len(all_data)-3):
        if (all_data['modified_swing_classes'][frame] == 1) and (all_data['modified_swing_classes'][frame+1] != 1) and (all_data['modified_swing_classes'][frame+2] == 0) and (all_data['modified_swing_classes'][frame+3] != 1):
            # forehand
            forehand_frames.append(frame)
        elif (all_data['modified_swing_classes'][frame] == 2) and (all_data['modified_swing_classes'][frame+1] != 2) and (all_data['modified_swing_classes'][frame+2] == 0) and (all_data['modified_swing_classes'][frame+3] != 2):
            #backhand
            backhand_frames.append(frame)
    return forehand_frames, backhand_frames

def get_region_classes(all_data, frames, fps, polygons):
    regions = []
    for frame in frames:
        try:
            region_class = predcit_shot_region(all_data, frame, polygons, fps)
            regions.append([frame, region_class])
        except IndexError:
            pass
    regions = np.array(regions, dtype=int)

    return regions


def predcit_shot_region(all_data, frame, polygons, fps):
    frame_range = [frame + int(fps*0), frame + int(fps*1)]
    x_points = all_data['x_ball'][frame_range[0]:frame_range[1]]
    y_points = all_data['y_ball'][frame_range[0]:frame_range[1]]
    points = np.array([x_points, y_points], dtype=int).transpose().tolist()
    region_classes = judge_region(points, polygons)
    region_classes = np.delete(region_classes, np.where(region_classes == 0))
    unique, freq = np.unique(region_classes, return_counts=True)
    try:
        region_class = unique[np.argmax(freq)]
    except ValueError:
        region_class = 0

    # print(frame, frame_range[0], region_classes, frame_range[1])
    # print(region_class)
    
    return region_class

def get_region_count(regions, region_class, player_position=None, all_data=None, tennis_court_width=823):
    try:
        if player_position is None:
            count = np.count_nonzero(regions[:, 1] == region_class)
        elif player_position == 'left':
            count = np.count_nonzero(regions[all_data['x_player_converted'][regions[:, 0]].values < tennis_court_width/2, 1] == region_class)
        elif player_position == 'right':
            count = np.count_nonzero(regions[all_data['x_player_converted'][regions[:, 0]].values >= tennis_court_width/2, 1] == region_class)
        else:
            raise('input valid player_position')
    except IndexError:
        count = 0
    
    return count

def get_region_ratio(regions, player_position, all_data):
    num_center = get_region_count(regions, 1, player_position, all_data)
    num_left = get_region_count(regions, 2, player_position, all_data)
    num_right = get_region_count(regions, 3, player_position, all_data)

    num_total = num_center + num_left + num_right 
    region_ratio = np.array([num_left, num_center, num_right])*100/num_total
    region_ratio = region_ratio.tolist()

    for i in range(3):
        if math.isnan(region_ratio[i]):
          region_ratio[i] = 'undetected'
        else: region_ratio[i] = int(region_ratio[i])
    return region_ratio