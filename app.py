
from flask import Flask, render_template, json, request, url_for, redirect
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)


@app.route('/')
def index():
	return render_template('index.html')