#!/usr/bin/python3

from flask import Flask, render_template, request, json, jsonify
from model import def_model
import datetime as dt
import keras
import h5py
import numpy as np

app = Flask(__name__)

model = def_model()
model.load_weights("model.h5")

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api', methods=['POST'])
def api():
    global model
    if not request.is_json:
        output = request.form['input'].strip()

    else:
        req_data = request.get_json()
        output = req_data['input'].strip()

    clean = np.array(output.split(",")).reshape(1, 13)

    result = model.predict(clean)

    return str(result.item(0))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
