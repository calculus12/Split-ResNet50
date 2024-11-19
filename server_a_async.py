# server_a.py

import io
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests
import importlib.util

import time
import threading
import queue
import json
import fcntl
import base64

# job queue
transmission_queue = queue.Queue()
# result_queue = queue.Queue(maxsize=50)


# Initialize Flask app
app = Flask(__name__)

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

# 컨테이너 정보
CONTAINER_B_HOST = os.environ.get('CONTAINER_B_HOST', 'localhost')
CONTAINER_B_PORT = os.environ.get('CONTAINER_B_PORT', '5001')
SHARED_FILE_PATH = os.environ.get('RESULT_FILE_PATH', '/shared/data_queue.json')

def add_data_to_file(data):
    # 파일이 없으면 생성하고 빈 리스트를 저장
    if not os.path.exists(SHARED_FILE_PATH):
        with open(SHARED_FILE_PATH, 'w') as f:
            json.dump([], f)

    # 파일 열기 및 잠금
    with open(SHARED_FILE_PATH, 'r+') as f:
        # 파일 잠금 설정 (쓰기 잠금)
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # 기존 데이터 로드
            try:
                f.seek(0)
                data_list = json.load(f)
            except json.JSONDecodeError:
                data_list = []

            # 새로운 데이터 추가
            data_list.append(data)

            # 파일 내용 업데이트
            f.seek(0)
            f.truncate()
            json.dump(data_list, f)
        finally:
            # 파일 잠금 해제
            fcntl.flock(f, fcntl.LOCK_UN)

def get_data_from_file():
    # 파일이 없으면 데이터가 없는 것임
    if not os.path.exists(SHARED_FILE_PATH):
        return None

    # 파일 열기 및 잠금
    with open(SHARED_FILE_PATH, 'r+') as f:
        # 파일 잠금 설정 (쓰기 잠금)
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # 데이터 로드
            try:
                f.seek(0)
                data_list = json.load(f)
            except json.JSONDecodeError:
                data_list = []

            if data_list:
                # 첫 번째 데이터 가져오기
                data = data_list.pop(0)

                # 파일 내용 업데이트
                f.seek(0)
                f.truncate()
                json.dump(data_list, f)

                return data
            else:
                return None
        finally:
            # 파일 잠금 해제
            fcntl.flock(f, fcntl.LOCK_UN)


def transmission_thread_func():
    while True:
        data, start_time, inference_time_a = transmission_queue.get()
        json_data = {
            'data': base64.b64encode(data).decode('utf-8'),
            'start_time': start_time,
        }
        try:
            # container_b 로 중간 추론 결과 전송
            response = requests.post(f"http://{CONTAINER_B_HOST}:{CONTAINER_B_PORT}/complete", json=json_data)
            response_data = response.json()
            response_data['inference_time_a'] = inference_time_a
            add_data_to_file(response_data)
            
        except Exception as e:
            response_data = {'error': str(e)}
        finally:
            # 작업 상태 업데이트
            transmission_queue.task_done()

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

# 전송 스레드 시작
transmission_thread = threading.Thread(target=transmission_thread_func, daemon=True)
transmission_thread.start()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['file']
        img_bytes = file.read()

        start_time = time.time()
        input_tensor = transform_image(img_bytes)
        # Get activation maps from Model A
        with torch.no_grad():
            activation_maps = model_a(input_tensor)

        # Serialize activation maps
        buffer = io.BytesIO()
        torch.save(activation_maps.cpu(), buffer)
        data = buffer.getvalue()
        inference_time_a = time.time() - start_time

        # 큐에 데이터 적재
        transmission_queue.put((data, start_time, inference_time_a))

        return jsonify({'success': True}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/poll', methods=['GET'])
def poll_data():

        # 큐에서 데이터를 꺼냄
    data = get_data_from_file()
    if data:
        return jsonify(data), 200
    else:
        return jsonify({'message': 'No data available'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
