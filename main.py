import os
from flask import Flask, render_template, url_for, flash, redirect, request
from werkzeug.utils import secure_filename
from generator_detector import validate_from_url, Generator_pattern_detection

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'py', 'zip'}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/github", methods = ['POST'])
def giturl():
    input = request.form['url']
    return validate_from_url(input)


@app.route('/scanfolder', methods = ['POST'])
def folder():
    input = request.form['folder']
    generator_detection = Generator_pattern_detection()
    return generator_detection.generator_detection(input)


@app.route('/uploadfile', methods = ['POST'])
def upload_file():
    f = request.files['file']
    f.save(os.path.join(os.getcwd(), 'downloaded_validate', secure_filename(f.filename)))
    generator_detection = Generator_pattern_detection()
    input = os.path.join(os.getcwd(), 'downloaded_validate', f.filename)
    return generator_detection.generator_detection(input)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)