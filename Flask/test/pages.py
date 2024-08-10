from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import json
import os

bp = Blueprint("pages", __name__)

@bp.route("/")
def home():
    return render_template("home.html")

@bp.route("/about")
def about():
    return render_template("about.html")

@bp.route("/prediction", methods = ['GET','POST']) #need the get method to allow the render_template to work

def prediction():
    if request.method == 'GET':
        return render_template("prediction.html")
    elif request.method == 'POST':
        data = request.get_json()
        svc_pipeline = pickle.load(open("test/Model/svc_pipeline.pkl", 'rb'))
        prediction = np.array2string(svc_pipeline.predict(data))
        return jsonify(prediction)