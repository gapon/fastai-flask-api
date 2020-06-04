from flask import Flask, jsonify
from fastai.vision import *

app = Flask(__name__)

path = Path('data')
defaults.device = torch.device('cpu')
learn = load_learner(path)

def get_prediction(img):
    pred_class,pred_idx,outputs = learn.predict(img)

    return str(pred_class), str(pred_idx.item()), str(outputs)

@app.route('/predict', methods=['GET'])
def predict():
    img = open_image(path/'test.jpg').resize(224)
    pred_class,pred_idx,outputs = get_prediction(img)

    return jsonify({'class_name': pred_class, 'class_id': pred_idx, 'outputs': outputs})