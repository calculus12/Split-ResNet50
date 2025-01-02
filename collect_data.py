import asyncio
import aiohttp
import time
import csv
import os
import socketio
import requests
import sys

# 서버의 엔드포인트 URL을 입력하세요
A_URL = ''
PROXY_URL = ''

IMAGE_PATH = ''

NUM_POD_A = 2
NUM_POD_B = 2
BANDWIDTH = 1

# asyncio.Future 객체 생성 (이벤트 발생을 대기)
event_future = asyncio.get_event_loop().create_future()

start_time = None
end_time = None




async def listen_socketio_event(sio_url, event_name):    
    """
    Socket.IO 클라이언트를 사용해 특정 이벤트를 대기합니다.
    """
    
    sio = socketio.AsyncClient()
    global end_time
    global start_time
    # N개 추론이 완료되면 시각 측정 후 요청 시각을 빼서 걸린 시간을 기록
    @sio.on('queue_threshold_reached')
    async def on_queue_threshold_reached(data):
        global end_time
        global start_time
        end_time = float(data['time']) - start_time
        await sio.disconnect()

    # @sio.event
    # async def connect():
    #     print("Socket.IO 서버에 연결되었습니다!")

    # @sio.event
    # async def disconnect():
    #     print("Socket.IO 서버와 연결이 끊어졌습니다.")

    # @sio.on(event_name)
    # async def handle_event(data):
    #     print(f"이벤트 수신: {event_name}, 데이터: {data}")
    #     await sio.disconnect()  # 이벤트 수신 후 연결 끊기

    await sio.connect(PROXY_URL)
    print(f"{event_name} 이벤트 대기 중...")
    await sio.wait()  # 이벤트 발생까지 대기

def ensure_directory_exists(dir_path):
    """
    디렉토리가 존재하지 않으면 생성합니다.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # 디렉토리 생성 (필요 시 부모 디렉토리도 생성)


def fetch_responses(url):
    try:
        response = requests.get(f'{url}/responses')
        response.raise_for_status()  # HTTP 오류가 발생하면 예외를 발생시킴
        data = response.json()  # JSON 데이터를 파싱하여 파이썬 객체로 변환
        return data
    except requests.exceptions.RequestException as e:
        print(f"서버로부터 데이터를 가져오는 중 오류 발생: {e}")
        return None
    
def compute_averages(data_list):
    if not data_list:
        print("데이터가 없습니다.")
        return None

    # 키 목록 가져오기 (모든 딕셔너리에 동일한 키가 있다고 가정)
    keys = data_list[0].keys()

    # 합계를 저장할 딕셔너리 초기화
    sums = {key: 0.0 for key in keys}

    # 각 항목의 값을 합산
    for data in data_list:
        for key in keys:
            sums[key] += data.get(key, 0.0)

    # 평균 계산
    count = len(data_list)
    averages = {key: sums[key] / count for key in keys}

    return averages



# 메인 함수
async def main(N, iterations, CSV_PATH):
    
    # 1 iteration마다 csv에 쓸 데이터
    results = []

    ensure_directory_exists(CSV_PATH)
    for _ in range(iterations):
        global start_time
        global end_time
        start_time = time.time()

        session = aiohttp.ClientSession()
        tasks = []
        for _ in range(N):
            with open(IMAGE_PATH, "rb") as image_file:
                image_data = image_file.read()
                # 요청 설정
                form = aiohttp.FormData()
                form.add_field('file', image_data, filename=IMAGE_PATH, content_type='image/jpeg')

                # 비동기 작업 추가
                tasks.append(session.post(A_URL, data=form))

        # 요청을 비동기로 실행
        for task in tasks:
            asyncio.create_task(task)  # 응답을 기다리지 않고 백그라운드에서 실행
        
        # 작업 완료 이벤트를 기다림 => 100개 추론이 완료될때까지 기다림
        await listen_socketio_event(PROXY_URL, 'queue_threshold_reached')
        await session.close()

        data_list = fetch_responses(PROXY_URL)

        inference_times_a = [x['inference_time_a'] for x in data_list]
        inference_times_b = [x['inference_time_b'] for x in data_list]
        total_time = [x['total_time'] for x in data_list]

        avg_a = sum(inference_times_a) / N
        avg_b = sum(inference_times_b)/ N
        avg_t = sum(total_time) / N

        results.append([avg_a, avg_b, avg_t, end_time, NUM_POD_A, NUM_POD_B, BANDWIDTH])
        
        # CSV 파일에 결과 저장
        with open(f'{CSV_PATH}/results_A{NUM_POD_A}_B{NUM_POD_B}_{BANDWIDTH}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['inference_time_a', 'inference_time_b', 'total_time', 'end_time', 'NUM_POD_A', 'NUM_POD_B', 'BANDWIDTH'])
            writer.writerows(results)
    
    # 프록시 서버의 응답 큐 초기화
    requests.post(f'{PROXY_URL}/clear_responses')


# 실행 부분
if __name__ == '__main__':
    BANDWIDTH = int(sys.argv[1])
    N = int(sys.argv[2])  # 동시에 보낼 요청의 수를 설정하세요
    CSV_PATH = os.path.join(os.getcwd(), f'./result1/bandwidth_{BANDWIDTH}')
    iterations = 5  # 반복 횟수를 설정하세요

    asyncio.run(main(N, iterations, CSV_PATH))