# server_a.py

import io
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests
# import pickle
# from model_definitions3 import ModelA
import importlib.util

import time

# Initialize Flask app
app = Flask(__name__)

# Load Model A
# model_a = ModelA()
# model_a.load_state_dict(torch.load('model_a.pth', map_location=torch.device('cpu')))
# model_a.eval()

# Read environment variables
MODEL_VERSION = os.environ.get('MODEL_VERSION', '1')  # Defaults to '1' if not set
# Dynamically import the model definitions module
model_definitions_module = f"model_definitions{MODEL_VERSION}"
spec = importlib.util.spec_from_file_location("model_definitions", f"{model_definitions_module}.py")
model_definitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_definitions)

# Load Model A
model_a = model_definitions.ModelA()
model_a_path = os.path.join('models', f"model_a{MODEL_VERSION}.pth")
model_a.load_state_dict(torch.load(model_a_path, map_location=torch.device('cpu')))
model_a.eval()

# Define image preprocessing
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['file']
        img_bytes = file.read()
        input_tensor = transform_image(img_bytes)

        
        # Get activation maps from Model A
        with torch.no_grad():
            activation_maps = model_a(input_tensor)
        inference_time_a = time.time() - start_time
        # Serialize activation maps
        # data = pickle.dumps(activation_maps.numpy())

        # Serialize activation maps
        buffer = io.BytesIO()
        torch.save(activation_maps.cpu(), buffer)
        data = buffer.getvalue()

        # Get Container B address from environment variable
        container_b_host = os.environ.get('CONTAINER_B_HOST', 'localhost')
        container_b_port = os.environ.get('CONTAINER_B_PORT', '5001')
        # Send data to Container B
        response = requests.post(f"http://{container_b_host}:{container_b_port}/complete", data=data)

        if response.status_code == 200:
            result = response.json()
            result['inference_time_a'] = inference_time_a
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to get response from Container B'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Replace '0.0.0.0' with your host if necessary
    app.run(host='0.0.0.0', port=5000)
