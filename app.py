from flask import Flask, request, jsonify, send_from_directory
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(244),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])
])

model = torchvision.models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 7)  # 假设有7个分类
model.load_state_dict(torch.load(r"E:\github\my_github\Flask_cat_7classify\best_model_train92.81.pth", map_location=device))
model.to(device)
model.eval()

categories = ['俄罗斯蓝猫', '孟买猫', '布偶猫', '暹罗猫', '波斯猫', '缅因猫', '英国短毛猫']

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img_bytes = file.read()
        try:
            prediction_index = predict_image(img_bytes)
            prediction_label = categories[prediction_index]
            return jsonify({'prediction': prediction_label})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('templates', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
