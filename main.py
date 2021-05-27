from flask import Flask, render_template, url_for, flash, redirect, request
from generator_detector import validate_from_url, Generator_pattern_detection


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/github", methods = ['POST', 'GET'])
def giturl():
    if request.method == 'POST':
        input = request.form['url']
    else:
        input = request.args['url']

    return validate_from_url(input)

@app.route('/scanfolder', methods = ['POST'])
def folder():
    input = request.form['folder']
    generator_detection = Generator_pattern_detection()
    return generator_detection.generator_detection(input)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)