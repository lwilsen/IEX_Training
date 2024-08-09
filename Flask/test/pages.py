from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import json

bp = Blueprint("pages", __name__)

@bp.route("/")
def home():
    return render_template("pages/home.html")

@bp.route("/about")
def about():
    return render_template("pages/about.html")

svc_pipeline = pickle.load(open("svc_pipeline.pkl", 'rb'))
