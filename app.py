from flask import Flask, jsonify, request, render_template
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


#@app.route('/')
#def hello_world():
#    return "Test Message"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img = get_image(img_bytes)
        pred_class,pred_idx,outputs = get_prediction(img)
        #return jsonify({'class_name': pred_class, 'class_id': pred_idx, 'outputs': outputs})
        return render_template('result.html', class_id=pred_idx, class_name=pred_class, outputs=outputs)
    return render_template('index.html')


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
    app.run(debug=True, host='0.0.0.0', port=8080)