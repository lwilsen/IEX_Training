from flask import Flask

from test import pages

import pickle as p

def create_app():
    app = Flask(__name__)

    app.register_blueprint(pages.bp)

    return app