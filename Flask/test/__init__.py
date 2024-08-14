from flask import Flask
from flask_cors import CORS #figure out how to get app to work with CORS
from test import pages

import pickle as p

def create_app():
    app = Flask(__name__)

    app.register_blueprint(pages.bp)

    return app
