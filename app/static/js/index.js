// 'use strict';

function main(img_path){
    // canvasの高さ、幅を指定して画像を表示
    img = new Image();
    img.onload = function(e){
        image_width  = img.width;
        image_height = img.height;
        element_width = Math.min(window.innerWidth*0.9, 800);
        element_height = element_width*(image_height/image_width);
        var canvas = document.getElementById('canvas');
        canvas.width = element_width;
        canvas.height = element_height;
        
        context = canvas.getContext('2d');
        context.drawImage(img, 0, 0, element_width, element_height);
    }
    img.src = img_path;

    // クリックされたら...
    coordinates = []
    canvas.addEventListener('click', Onclick, false);
}

function Onclick(e) {
    
    var r = canvas.getBoundingClientRect();
    var x_element = Math.max(Math.round(event.clientX-r.left), 0);
    var y_element = Math.max(Math.round(event.clientY-r.top), 0);
    var x_image = Math.round(x_element * (image_width / element_width));
    var y_image = Math.round(y_element * (image_height / element_height));

    // リストに保存
    coordinates.push([x_element, y_element, x_image, y_image]);

    if (coordinates.length <= 4){
        // 点を描画
        draw_court(coordinates);
    }
    else {
        alert("You cannot enter more than 4 dots.");
        coordinates.pop();
    }
}

function draw_court(coordinates){
    // 点の描画
    for(var i=0; i<coordinates.length; i++){
        x = coordinates[i][0];
        y = coordinates[i][1];
        context.fillRect(x-3 ,y-3 ,7, 7);
    }
    // 線の描画
    if (coordinates.length >= 1){
        for(var i=0; i<coordinates.length-1; i++){
            context.beginPath();
            context.moveTo(coordinates[i][0], coordinates[i][1]);
            context.lineTo(coordinates[i+1][0], coordinates[i+1][1]);
            context.stroke();
        }
        if (coordinates.length == 4){
            context.beginPath();
            context.moveTo(coordinates[3][0], coordinates[3][1]);
            context.lineTo(coordinates[0][0], coordinates[0][1]);
            context.stroke();
        }
    }
}

function undo() {
    context.drawImage(img, 0, 0, element_width, element_height);
    if (coordinates.length >= 0){
    coordinates.pop();
    }
    draw_court(coordinates);
}

function send_court() {
    if (coordinates.length == 4){
        court_points = {court_point_1: [coordinates[0][2], coordinates[0][3]],
                        court_point_2: [coordinates[1][2], coordinates[1][3]],
                        court_point_3: [coordinates[2][2], coordinates[2][3]],
                        court_point_4: [coordinates[3][2], coordinates[3][3]]};
        json = JSON.stringify(court_points);
        $.ajax({
            type: "POST",
            url: "/result",
            data: json,
            contentType: "application/json",
            success: function(msg) {
                console.log(msg);
            },
            error: function(msg) {
                console.log("error");
            }
        });
        
    }
}

function cancel_submit(){
    if (coordinates.length != 4){
        alert('Click on the four dots for single courts');
        return false;
    }
}

function loading(){
    // TODO メッセージを引数にとって、表示する機能
    $('#loading').show();
    $('#content').hide();       
}

function show_filename(){  
    var file = $('#input_movie_file').prop('files')[0];
    $('#filename').text(file.name + '     is selected');
}

// function 