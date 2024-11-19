# server_b.py

import os
import torch
from flask import Flask, request, jsonify
import pickle
# from model_definitions3 import ModelB
import io
import importlib.util

import time

# Initialize Flask app
app = Flask(__name__)

# Load Model B
# model_b = ModelB()
# model_b.load_state_dict(torch.load('model_b.pth', map_location=torch.device('cpu')))
# model_b.eval()

# Read environment variables
MODEL_VERSION = os.environ.get('MODEL_VERSION', '1')  # Defaults to '1' if not set

# Dynamically import the model definitions module
model_definitions_module = f"model_definitions{MODEL_VERSION}"
spec = importlib.util.spec_from_file_location("model_definitions", f"{model_definitions_module}.py")
model_definitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_definitions)

# Load Model B
model_b = model_definitions.ModelB()
model_b_path = os.path.join('models', f"model_b{MODEL_VERSION}.pth")
model_b.load_state_dict(torch.load(model_b_path, map_location=torch.device('cpu')))
model_b.eval()

# Load class labels
with open('imagenet_classes.txt') as f:
    idx_to_labels = [line.strip() for line in f.readlines()]

@app.route('/complete', methods=['POST'])
def complete():
    try:
        # data = request.data
        # activation_maps = pickle.loads(data)
        # activation_maps = torch.tensor(activation_maps)
        
        data = request.data
        buffer = io.BytesIO(data)
        activation_maps = torch.load(buffer)

        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_b(activation_maps)
            _, predicted = outputs.max(1)
            class_id = predicted.item()
            class_name = idx_to_labels[class_id]
        inference_time_b = time.time() - start_time

        return jsonify({'class_id': class_id, 'class_name': class_name, 'infernce_time_b': inference_time_b})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Replace '0.0.0.0' with your host if necessary
    app.run(host='0.0.0.0', port=5001)
