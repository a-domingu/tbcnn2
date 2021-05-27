from flask import Flask, render_template, url_for, flash, redirect, request
from forms import RegistrationForm, LoginForm
import sys
import os
import gensim
import random
import numpy
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import shutil
from generator_detector import validate_from_url, Generator_pattern_detection

from node_object_creator import *
from embeddings import Embedding
from node import Node
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer
from main_first_neural_network import set_vector, set_leaves
from repos import download_repos


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]


@app.route("/")
@app.route("/folderform")
def home():
    #return render_template('home.html', posts=posts)
    return """<form action="/scanfolder" method="post">
                Please enter a folder name: <input type="text" name="folder">
                <input type="submit" value="Scan">
              </form>"""

@app.route("/githubform")
def about():
    return """<form action="/github" method="post">
                Please enter a github repository url: <input type="text" name="url">
                <input type="submit" value="Scan">
              </form>"""


@app.route("/github", methods = ['POST'])
def giturl():
    input = request.form['url']

    return validate_from_url(input)

@app.route('/scanfolder', methods = ['POST'])
def folder():
    input = request.form['folder']
    generator_detection = Generator_pattern_detection()
    
    return generator_detection.generator_detection(input)

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)
