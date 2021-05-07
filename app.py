from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from text_input import custom_text_check
from emotion_detection import custom_audio_check
from model import test_one_file
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = "static/images"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def man():
    return render_template("index.html")


@app.route('/text/', methods=['POST', 'GET'])
def textData():
    text = request.form['param']
    result = custom_text_check(text)
    result = text + ": " + result
    return render_template("index.html", data=result)


@app.route('/audio/', methods=['POST', 'GET'])
def audioData():
    audio = request.files['avatar']
    audio.save(secure_filename(audio.filename))
    print(audio.filename)
    result = custom_audio_check()
    os.remove(audio.filename)
    result = audio.filename + ": " + result
    return render_template("index.html", data1=result)


@app.route('/image/', methods=['POST', 'GET'])
def imageData():
    if os.path.exists("static/images/detected1.jpg"):
        os.remove("static/images/detected1.jpg")
    image = request.files['avatar1']
    if image.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    image.save(secure_filename("orignal.jpg"))
    print(image.filename)
    result = test_one_file()
    os.remove("orignal.jpg")
    return render_template("index.html", data2=result)


@app.route('/display1/')
def display1_image():
    return redirect(url_for('static', filename='images/detected1.jpg'))


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == "__main__":
    app.run(debug=True)
