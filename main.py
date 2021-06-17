import os
import importlib
import zipfile
import shutil
from flask import Flask, render_template, url_for, flash, redirect, request
from werkzeug.utils import secure_filename
from pattern_detector import validate_from_url, Pattern_detection

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'py', 'zip'}
PATTERNS = ['generator', 'wrapper']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/github", methods = ['POST', 'GET'])
def giturl():
    output = ''
    for pattern in PATTERNS:
        if request.method == 'POST':
            input = request.form['url']
        else:
            input = request.args['url']
        output = output + validate_from_url(input, pattern)
    
    return output


@app.route('/scanfolder', methods = ['GET', 'POST'])
def scan_folder():
    if request.method == 'POST':
        f = request.files['folder']
        f.save(os.path.join(os.getcwd(), 'downloaded_validate', secure_filename(f.filename)))

        if f.filename.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(os.getcwd(), 'downloaded_validate', f.filename), 'r') as zip_ref:
                os.mkdir(os.path.join(os.getcwd(), 'zipfolder', f.filename))
                zip_ref.extractall(os.path.join(os.getcwd(), 'zipfolder', f.filename))

            remove_folder(os.path.join(os.getcwd(), 'downloaded_validate'))
            input = os.path.join('zipfolder', f.filename)
            output = ''
            for pattern in PATTERNS:
                class_name = pattern.capitalize() + '_detection'
                module = importlib.import_module('pattern_detection.' + pattern + '_detector')
                pattern_class = getattr(module, class_name)
                pattern_detector = pattern_class(pattern)
                output = output + pattern_detector.pattern_detection(input)
            remove_folder(os.path.join(os.getcwd(), 'zipfolder'))

        elif f.filename.endswith('.py'):  
            input = os.path.join('downloaded_validate', f.filename)
            output = ''    
            for pattern in PATTERNS:
                class_name = pattern.capitalize() + '_detection'
                module = importlib.import_module('pattern_detection.' + pattern + '_detector')
                pattern_class = getattr(module, class_name)
                pattern_detector = pattern_class(pattern)
                output = output + pattern_detector.pattern_detection(input) 
            os.remove(os.path.join(os.getcwd(), input))

        else:
            output = 'Invalid file extension. Please upload a Python or a zip file.' 

        return output


def remove_folder(dir):
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)

'''
@app.route('/uploadfile', methods = ['POST', 'GET'])
def scan_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(os.getcwd(), 'downloaded_validate', secure_filename(f.filename)))
        generator_detection = Generator_pattern_detection()
        input = os.path.join('downloaded_validate', f.filename) 
        output = generator_detection.generator_detection(input)   
        os.remove(os.path.join(os.getcwd(), input)) 
        return output
'''

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)