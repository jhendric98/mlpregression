#!/usr/bin/python3

from flask import Flask, render_template, request, json, jsonify
from model import def_model
import datetime as dt

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/<string:data>')
def api(data):
    # req_data = request.get_json()
    # output = req_data['input'].strip()
    # return jsonify(output)
    return data


if __name__ == '__main__':
    app.run(debug=True)
