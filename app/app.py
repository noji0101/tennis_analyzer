import datetime
import os
import sys
import pathlib
import random
import json

from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import pandas as pd
import cv2
from cv2 import VideoWriter_fourcc

import app_utils
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from main import main
from modules.analyzer.plot_shot import plot_player_and_shot
from modules.analyzer.tendancy import tendancy


app = Flask(__name__)
app.secret_key = 'movie_path'
app.secret_key = 'movie_fps'
app.secret_key = 'num_total_frame'
app.secret_key = 'width'
app.secret_key = 'height'
app.secret_key = 'region_ratios'


UPLOAD_FOLDER = './app/static/uploads'
ALLOWED_EXTENSIONS = ['mp4']
DATAFRAME_PATH = os.path.join(UPLOAD_FOLDER, 'result', 'all_data.csv')
RESULT_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'result', 'player_and_shot.png')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_court', methods=['GET', 'POST'])
def select_court():
    if request.method == 'POST':
        input_movie_file = request.files['input_movie_file']
        if input_movie_file and app_utils.is_alowed_file(input_movie_file.filename, ALLOWED_EXTENSIONS):
            # 動画を保存
            filename = secure_filename(input_movie_file.filename)
            movie_path = os.path.join(UPLOAD_FOLDER, 'movie' ,filename)
            session['movie_path'] = movie_path
            input_movie_file.save(movie_path)

            # 動画からランダムで一枚を切り出して保存
            movie = cv2.VideoCapture(movie_path)
            num_total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
            movie_fps = movie.get(cv2.CAP_PROP_FPS)
            session['num_total_frame'] = num_total_frame
            session['movie_fps'] = movie_fps
            session['width'] = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
            session['height'] = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            sample_image_frame = random.randint(0, num_total_frame)
            frame_path = os.path.join(UPLOAD_FOLDER, 'movie', filename[:-4] + '.jpg')
            app_utils.save_frame(movie, sample_image_frame, frame_path)
                        
            input_img_url = './static/uploads/movie/' + filename[:-4] + '.jpg'
            return render_template('index.html', input_img_url=input_img_url)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for('/'))


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        court_points = request.json
        session['court_points'] = court_points
        print('court_pointsをセッションに保存')
        return '<p>loading...</p>'
    elif request.method == 'GET':
        if session.get('court_points') is not None:
            print('court_pointsを読み込み')
            config_file = 'config/config.yaml'
            frame_range = [0, int(session.get('num_total_frame'))]
            movie_path = session.get('movie_path', None)
            all_data = main(movie_path, config_file ,session['court_points'])
            all_data.to_csv(DATAFRAME_PATH)
            plot_player_and_shot(RESULT_IMAGE_PATH, all_data, frame_range=frame_range)
            # delete './app/'
            load_img_path = RESULT_IMAGE_PATH[6:]

            # get shot tendancy
            court_points = session.get('court_points', None)
            fps = session.get('movie_fps', None)
            video_width = session.get('width', None)
            region_ratios = tendancy(all_data, court_points, fps, video_width)
            print(region_ratios)
            session['region_ratios'] = region_ratios

            checked_list = ['checked' for _ in range(8)]
            return render_template('result.html', load_img_path=load_img_path, checked_list=checked_list, frame_range=frame_range, region_ratios=region_ratios)
    else:
        return redirect(url_for('/'))

@app.route('/result_change_image', methods=['GET', 'POST'])
def change_image():
    if request.method == 'POST':
        display_swing_indices = [int(i) for i in request.form.getlist('check')]
        checked_list = ['' for _ in range(8)]
        for index in display_swing_indices:
            checked_list[index] = 'checked'
        frame_range = [int(i) for i in request.form.getlist('frame')]
        all_data = pd.read_csv(DATAFRAME_PATH)

        # get result_image
        plot_player_and_shot(RESULT_IMAGE_PATH, all_data, frame_range=frame_range, display_swing_indices=display_swing_indices)
        # delete './app/'
        load_img_path = RESULT_IMAGE_PATH[6:]

        # get shot tendancy
        # region_ratios = session.get('region_ratios', None)
        court_points = session.get('court_points', None)
        fps = session.get('movie_fps', None)
        video_width = session.get('width', None)
        region_ratios = tendancy(all_data, court_points, fps, video_width)
        print(region_ratios)

        return render_template('result.html', load_img_path=load_img_path, checked_list=checked_list, frame_range=frame_range, region_ratios=region_ratios)
            

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=88, debug=True)