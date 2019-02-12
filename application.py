from flask import Flask
from flask import render_template
import register
import auth
import json
from flask import request, json


app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/train", methods = ['POST'])
def train():
    data = json.loads(request.data)
    register.register_local_image(data['filename'], data['identityname'])
    return json.dumps({'status': data['identityname'] + ' - Successfully Registered...'})

    
@app.route("/recognize", methods = ['POST'])
def recognize():
    data = request.get_json()
    name = auth.run_face_recognition_image_local(data['filename'])
    return json.dumps({'name':name})