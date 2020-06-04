from flask import Flask, jsonify, request
from fastai.vision import *
import io

app = Flask(__name__)

path = Path('data')
defaults.device = torch.device('cpu')
# Load model
learn = load_learner(path)

def get_image(image_bytes):
    return open_image(io.BytesIO(image_bytes)).resize(224)

def get_prediction(img):
    pred_class,pred_idx,outputs = learn.predict(img)
    return str(pred_class), str(pred_idx.item()), str(outputs)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get the file from the request
        file = request.files['file']
        img_bytes = file.read()
        img = get_image(img_bytes)
        pred_class,pred_idx,outputs = get_prediction(img)
        return jsonify({'class_name': pred_class, 'class_id': pred_idx, 'outputs': outputs})


if __name__ == '__main__':
    app.run()