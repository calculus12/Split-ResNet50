#!/bin/bash

if [ -z "$1" ]; then
    echo "사용법: $0 <데이터 개수>"
    exit 1
fi

# 서버 URL
SERVER_URL=""
DATA_NUM=$1

curl -X POST "$SERVER_URL/clear_responses"

curl -X POST "$SERVER_URL/set_queue_threshold" \
    -H "Content-Type: application/json" \
    -d "{\"queue_threshold\": \"$DATA_NUM\"}"

# 반복문: x = 1 ~ 8
for x in {1..8}; do
    echo "현재 x 값: $x"

    # 1. 변수 x를 사용하여 curl POST 요청
    curl -X POST "$SERVER_URL/set_threads" \
         -H "Content-Type: application/json" \
         -d "{\"thread_count\": \"$x\"}"
    echo "스레드 설정 완료 (x=$x)"

    # 2. 변수 x를 인자로 Python 스크립트 실행
    python collect_data.py "$x" "$DATA_NUM"
    echo "collec_data.py 스크립트 실행 완료(x=$x)"
done

