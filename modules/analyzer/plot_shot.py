import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import cv2
import numpy as np

def plot_player_and_shot(save_path, all_data, frame_range=None, display_swing_indices=None, swing_classes='modified_swing_classes'):

    if frame_range is None:
        start, stop = 0, len(all_data)
    else:
        start, stop = frame_range[0], frame_range[1]
    x_player = all_data['x_player_converted'].tolist()[start:stop]
    y_player = all_data['y_player_converted'].tolist()[start:stop]

    if display_swing_indices is None:
        colors = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow", 'orange']
    else:
        colors = ["white" for _ in range(8)]
        default_colors =  ["gray", "red", "green", "blue", "cyan", "magenta", "yellow", 'orange']
        for index in display_swing_indices:
            colors[index] = default_colors[index]
    cmap = ListedColormap(colors)
    swing_classes = all_data[swing_classes].tolist()[start:stop]

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    ax = fig.add_subplot(1,1,1)
    ax.invert_yaxis()
    ax.set_xlim([-700, 1600])
    ax.set_ylim([1500, -100])
    ax.set_aspect('equal')
    ax.scatter(x_player, y_player, c=swing_classes, cmap=cmap, s=1, alpha=0.5)
    # draw court
    r = patches.Rectangle(xy=(0, 0), width=823, height=1280, ec='#000000', fill=False)
    ax.add_patch(r)
    fig.savefig(save_path)
