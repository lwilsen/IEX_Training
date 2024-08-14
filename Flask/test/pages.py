from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, g
import numpy as np
import pickle
import json
import sqlite3

bp = Blueprint("pages", __name__)

database = "/Users/lukewilsen/Desktop/IEX/IEX_Training/sqlite/titanic.db"

@bp.route("/")
def home():
    return render_template("home.html")

@bp.route("/about")
def about():
    return render_template("about.html")

@bp.route("/prediction", methods = ['GET','POST'])
def prediction():
    if request.method == 'GET':
        return render_template("prediction.html")
    elif request.method == 'POST':
        data = request.get_json()
        svc_pipeline = pickle.load(open("test/Model/svc_pipeline.pkl", 'rb'))
        prediction = np.array2string(svc_pipeline.predict(data))
        return jsonify(prediction)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(database)
    return db

@bp.route("/index")
def index():
    cur = get_db().cursor()
    cur.execute("SELECT * FROM train LIMIT 5")
    rows = cur.fetchall()
    return render_template("data_table.html",rows = rows)

@bp.teardown_request
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()