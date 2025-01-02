from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import queue
import requests
import os
import time

app = Flask(__name__)
socketio = SocketIO(app)

# 전송 스레드 수를 저장하는 변수
transfer_thread_count = 1  # 기본값은 1

# 전송 작업 큐 (서버 A로부터 받은 데이터)
transfer_queue = queue.Queue()

# 응답 데이터 큐 (서버 B로부터 받은 응답 데이터)
response_queue = queue.Queue()

# 전송 스레드 목록
transfer_threads = []

# 큐 특정 개수
queue_threshold = 20

# 전송 스레드 실행 플래그
transfer_thread_running = threading.Event()
transfer_thread_running.set()


CONTAINER_B_HOST = os.environ.get('CONTAINER_B_HOST', 'localhost')
CONTAINER_B_PORT = os.environ.get('CONTAINER_B_PORT', '5001')
SLEEP_TIME = os.environ.get('SLEEP_TIME', '')

# 스레드 종료를 위한 Sentinel Value 정의
SENTINEL = object()
# 전송 스레드 함수
def transfer_worker():
    while True:
        data = transfer_queue.get()
        
        if data is SENTINEL:
            # print('sentinel!')
            # 스레드 종료
            transfer_queue.task_done()
            break
        try:
            # print('get data!')
            # 서버 B로 데이터 전송
            if bool(SLEEP_TIME):
                time.sleep(float(SLEEP_TIME))
            response = requests.post(f"http://{CONTAINER_B_HOST}:{CONTAINER_B_PORT}/complete", 
                            json=data)
            response_data = response.json()
            # 응답 큐에 데이터 저장
            response_data['inference_time_a'] = data.get('inference_time_a')
            response_queue.put(response_data)
            if response_queue.qsize() >= queue_threshold:
                # 클라이언트로 이벤트 전송
                socketio.emit('queue_threshold_reached', {'time': time.time()})
        except Exception as e:
            # 에러 발생 시 응답 큐에 에러 정보 저장
            response_queue.put({'error': str(e)})
        finally:
            transfer_queue.task_done()


# 전송 스레드 시작 함수
def start_transfer_threads():
    global transfer_threads
    stop_transfer_threads()  # 기존 스레드 중지
    transfer_threads = []
    for _ in range(transfer_thread_count):
        t = threading.Thread(target=transfer_worker, daemon=True)
        t.start()
        transfer_threads.append(t)

# 전송 스레드 중지 함수
def stop_transfer_threads():
    global transfer_threads
    i = 0
    # 각 스레드 수만큼 Sentinel Value를 큐에 넣어서 스레드 종료 신호 전송
    for _ in transfer_threads:
        i += 1
        transfer_queue.put(SENTINEL)
    # 모든 스레드가 종료될 때까지 기다림
    # print (i, 'SENTINEL added')
    for t in transfer_threads:
        t.join()
    transfer_threads = []

# 초기 전송 스레드 시작
start_transfer_threads()

# 서버 A로부터 데이터 수신 엔드포인트
@app.route('/complete', methods=['POST'])
def receive_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    # 데이터 작업 큐에 추가
    transfer_queue.put(data)

    return jsonify({'message': 'Data received and added to transfer queue'}), 200

# 응답 데이터 큐 반환 엔드포인트
@app.route('/responses', methods=['GET'])
def get_responses():
    responses = []
    while not response_queue.empty():
        responses.append(response_queue.get())
    return jsonify(responses), 200

# 전송 스레드 개수 설정 엔드포인트
@app.route('/set_threads', methods=['POST'])
def set_threads():
    global transfer_thread_count
    data = request.get_json()
    if not data or 'thread_count' not in data:
        return jsonify({'error': 'No thread_count provided'}), 400
    try:
        thread_count = int(data['thread_count'])
        if thread_count <= 0:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Invalid thread_count value'}), 400
    transfer_thread_count = thread_count
    start_transfer_threads()
    return jsonify({'message': f'Transfer thread count set to {transfer_thread_count}'}), 200

# 응답 데이터 큐 초기화 엔드포인트
@app.route('/clear_responses', methods=['POST'])
def clear_responses():
    with response_queue.mutex:
        response_queue.queue.clear()
    return jsonify({'message': 'Response queue cleared'}), 200

# 큐 임계값 설정
@app.route('/set_queue_threshold', methods=['POST'])
def set_queue_threshold():
    data = request.get_json()
    if not data or 'queue_threshold' not in data:
        return jsonify({'error': 'No queue_threshold provided'}), 400
    try:
        threshold = int(data['queue_threshold'])
        if threshold <= 0:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Invalid thread_count value'}), 400
    global queue_threshold
    queue_threshold = threshold
    return jsonify({'message': f'Queue threshold count set to {queue_threshold}'}), 200


# 응답 데이터 큐의 데이터 개수 반환 엔드포인트
@app.route('/response_count', methods=['GET'])
def response_count():
    count = response_queue.qsize()
    return jsonify({'count': count}), 200

# 요청 데이터 큐의 데이터 개수 반환 엔드포인트
@app.route('/request_count', methods=['GET'])
def request_count():
    count = transfer_queue.qsize()
    return jsonify({'count': count}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=6000, allow_unsafe_werkzeug=True)
