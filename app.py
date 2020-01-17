from sqlalchemy import func
import pandas as pd

from flask import (
    Flask,
    render_template,
    jsonify,
    make_response)

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


import csv
from flask_cors import (CORS , cross_origin)
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
db_url = "postgresql://postgres:Xjy921513!@localhost:5432/collisionsdata"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
cors = CORS(app, resources={r"/*": {"origins": "*"}})
db = SQLAlchemy(app)


"""class Bigfoot(db.Model):
    __tablename__ = 'bigfoot'

    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer)
    title = db.Column(db.String)
    classification = db.Column(db.String)
    timestamp = db.Column(db.String)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    def __repr__(self):
        return '<BigFoot %r>' % (self.name)
"""

#default port number is 5000
#URL for flask: localhost:5000
#

@app.route('localhost:5000', methods=['POST'])
def get_names():
   if request.method == 'POST':
       data = request.get_json()

       return data



# Create database classes
@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()

    print('here')


@app.route("/")
def home():
    """Render Home Page."""

    return render_template("index.html")


@app.route("/index.html")
def index():
    """Render Home Page."""

    return render_template("index.html")






