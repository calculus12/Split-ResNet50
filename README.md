## 프로젝트 개요
1. 쿠버네티스를 활용한 분산 환경에서 딥러닝 모델을 두 개로 분할하고 각 노드에서 서빙한다.
2. 분산된 노드 사이에서의 트래픽에 따라 파드 개수를 조절함으로써 (horizontal scailing) 노드의 리소스를 효율적으로 활용

중간에 트래픽이 증가하면 네트워크의 병목으로 인해 파드의 개수를 늘려 추론 처리량을 늘려도 효용이 없다 -> 따라서 파드의 개수를 감소시킨다.
<img width="643" alt="trafficup" src="https://github.com/user-attachments/assets/49c27217-7c70-4262-9424-64467be09274" />

중간에 트래픽이 감소하면 네트워크의 병목이 줄어들어 파드의 개수를 늘려 추론 처리량을 알맞게 증가시킨다. -> 따라서 파드의 개수를 증가시킨다.
<img width="643" alt="trafficdown" src="https://github.com/user-attachments/assets/2921fc71-e470-4a4c-ba0c-df350bd69317" />

이러한 스케일링을 결정해줄 오토 스케일링 모델 만들기

## 모델 분할
- imagenet 1k 데이터셋으로 pretrained 된 resnet-50 모델을 분할
- 분할된 모델들이 residual block에 의한 의존성이 없게끔 분할한다.
- FLOPS와 weight 개수에 따라 다양하게 분할
  - `model_definitions1.py`, `model_definitions2.py`, `model_definitions3.py`
  - 나뉘는 부분을 A, B라고 했을때 1에서 3으로 갈 수록 A의 비중이 작아지는 버전
  - 버전 2가 가장 균등하게 나눠진 모델

<img width="400" alt="model split" src="https://github.com/user-attachments/assets/f1c82ebd-7e8e-4aa0-9ed3-0e5218f30a08" />


## 클러스터 모델 서버 및 프록시
두 개의 워커 노드와 하나의 마스터 노드로 구성된 클러스터에서 위의 분할된 모델을 서빙

<img width="645" alt="cluster" src="https://github.com/user-attachments/assets/4a8a7e98-ee24-42b5-83ab-04ea73b0964d" />


- `server_a.py`, `server_b.py` 가 각각 워커노드 1과 워커노드 2에서 모델을 서빙한다.
  - 모델 분할 버전과 프록시의 호스트, 포트, 다른 워커노드에서 서빙되는 서버의 호스트와 포트 등 환경변수로 입력 가능
- `Dockerflie_A.py`, `Dockerfile_B.py` 를 통해 도커 컨테이너 이미지로 생성
- `proxy.py` 는 워커노드 1과 워커노드 2와 중간에서, 워커노드 1로부터 받은 데이터를 그대로 워커노드 2로 포워딩
  -  이때 프록시의 스레드 개수를 조정함으로써 동시에 보내는 개수를 조절하여 트래픽을 에뮬레이션
  -  스레드 개수가 적으면 트래픽이 큰 것을 에뮬레이션 하고, 스레드 개수가 크면 트래픽이 원할한 것을 에뮬레이션
<img width="445" alt="proxy" src="https://github.com/user-attachments/assets/32cd0a41-7dd4-40c8-a9d7-530226bfe71e" />


- 이미지를 전송하여 테스트하기 위해서는 클러스터의 워커 노드1은 NodePort로 열려있어야 한다.
  - `server_a.py`의 /predict 혹은 /predict-proxy 엔드포인트로 이미지를 전송하면 된다.
  - /predict 엔드포인트는 프록시를 거치지 않고 워커노드 2로 바로 중간 추론 결과를 전달
- `proxy.py`는 워커 노드2로 포워딩 한 후에 추론 결과를 `response_queue`에 저장
  - 이때 테스트 및 데이터 수집 목적으로 위 큐의 응답 데이터가 일정 개수 이상 쌓이면 웹소켓으로 연결된 클라이언트(`collect_data.py`)에 알람을 보냄 (e.g. 요청을 100개 보내고 추론이 다 끝난 순간 사이의 시간을 측정하기 위해)
  - /set_threads 엔드포인트로 프록시의 스레드 개수 설정 가능

## 트래픽에 따라 파드의 개수를 조정해 줄 오토 스케일링 에이전트 만들기

1. 먼저 파드 개수와 트래픽 (프록시의 스레드 개수)에 따른 걸리는 추론시간을 측정
   - `collect_data.py`와 `collect_data.sh` 을 활용
2. 수집한 데이터가 적으면 위에서 얻은 데이터를 exponential non-linear regression으로 Q-learning에 사용할 환경을 인위적으로 만든다.
3. 아래의 hyperparameters 를 조정해가면서 Q-learning 에이전트를 생성 (`Q-learning.ipynb`)
<img width="373" alt="q-learning" src="https://github.com/user-attachments/assets/0b5cd0c0-dd5a-408d-8b94-1979594b256a" />
