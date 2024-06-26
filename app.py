from flask import Flask, jsonify, request
import pickle
import numpy as np
import os as os   
import random 
import cv2
import imutils
import random
from sklearn.preprocessing import LabelBinarizer 
from werkzeug.utils import secure_filename
from flask_cors import CORS
import re
import joblib


app = Flask(__name__)
CORS(app)

pickled_model = pickle.load(open('model.pkl', 'rb'))

with open('./models/transform_predict.joblib', 'rb') as f:
    model = joblib.load(f)

@app.route('/getdyslexic', methods=['POST'])
def create_book2():
    try:
        json_data = request.json
        if json_data and 'data' in json_data:
            data_list = json_data['data']
        latest_data = np.array([float(data_list["language_vocab"]), float(data_list["visual_discrimination"]),
                                float(data_list["memory"]), float(data_list["audio_Discrimination"]), float(np.random.uniform(0, 0.5)),0.2])
        latest_data = latest_data.reshape(1, -1)

        prediction = float(model.predict(latest_data)[0])
        response_data = {'result': prediction}
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

with open('label_binarizer.pkl', 'rb') as f:
    LB = pickle.load(f)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)
        ypred = pickled_model.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word

@app.route('/gettext', methods=['POST'])
def create_book():
    try:
        print(request.files)
        
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print(f.filename)
        letter, image = get_letters(f.filename)
        word = get_word(letter)
        pattern = r'\(([^)]+)\)'
        letters_in_parentheses = re.findall(pattern, word)
        response_data = {'word': letters_in_parentheses}
        print(word)
        return jsonify(response_data), 200

        print('hello2')

        return jsonify({'error': 'File upload failed'}), 500
    
    except Exception as e:
        print('hello')
        return jsonify({'error': str(e)}), 500
    
    # try:
    #     json_data = request.json
    #     if json_data and 'data' in json_data:
    #         data_list = json_data['data']

    #     letter,image = get_letters("cat-2.png")
    #     word = get_word(letter)
    #     response_data = {'word': word}
    #     return jsonify(response_data),200
    # except Exception as e:
    #    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)