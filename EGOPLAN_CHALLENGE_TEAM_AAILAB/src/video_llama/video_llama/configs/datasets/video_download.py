import csv
import os
import requests
import json

# 함수 정의: 동영상 다운로드 및 저장
def download_video(content_url, save_path):
    try:
        with requests.get(content_url, stream=True) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 오류가 발생하여 동영상을 다운로드할 수 없습니다: {e}")

json_file = 'videochat_instruct_11k.json'

# JSON 파일 읽기
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

video_list = []
for sample in data:
    # "video" 키에 대한 값 출력
    video_list.append(sample['video'])

# CSV 파일 경로 및 저장 디렉토리 지정
for i in range(7152):
    if i < 6230:
        continue
    name = f"{i:04d}"
    csv_file = f'./webvid-10M/data/train/partitions/{name}.csv'
    save_directory = './webvid-10M/data/videos'
    os.makedirs(save_directory,exist_ok=True)
    # print(csv_file)
    # assert 0

    # CSV 파일 읽기 및 처리
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        print(csv_file)
        reader = csv.DictReader(csvfile)
        for row in reader:
            videoid = row['videoid']
            contentUrl = row['contentUrl']
            page_dir = row['page_dir']
            # 파일 이름 생성
            filename = f"{page_dir}/{videoid}.mp4"
            # 저장할 경로 생성
            os.makedirs(os.path.join(save_directory, page_dir), exist_ok=True)
            save_path = os.path.join(save_directory, filename)
            if filename in video_list:
                # 동영상 다운로드 및 저장
                if os.path.exists(save_path):
                    # print(f"이미 존재하는 파일: {save_path}")
                    continue
                # 동영상 다운로드 및 저장
                download_video(contentUrl, save_path)
                print(f"다운로드 완료: {save_path}")
