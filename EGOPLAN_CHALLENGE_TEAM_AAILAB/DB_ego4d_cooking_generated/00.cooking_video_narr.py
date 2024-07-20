import json
import pandas as pd
import os 

#################### video_id_list: 내가 csv를 생성하고 싶은 video_id가 담긴 list ####################

video_id_file = pd.read_csv('./cooking_id_list.csv')
print(video_id_file['video_uid'])
video_id_list = list(video_id_file['video_uid'])

#################### valid_test에 있는 video만 json 따로 저장 ####################

# Ego4D dataset의 모든 video에 대한 narration이 저장된 json file 
file_path = './narration.json'

with open(file_path, 'r') as file:
    json_data = json.load(file)

extracted_data = {key: json_data[key] for key in video_id_list if key in json_data}
    
#################### csv로 바꾸기 ####################

def sort_by_timestamp_frame(dict_list):
    return sorted(dict_list, key=lambda x: x['timestamp_frame'])

final_json = {}

# 문제있는 video를 제외한 나머지 video만 저장
for key in extracted_data.keys():
  if (extracted_data[key]['status'] != 'complete' and 'narration_pass_2' not in extracted_data[key].keys()):
    continue
  if len(extracted_data[key]['narration_pass_2']['narrations']) == 0:
    continue
  else:
    final_narr = extracted_data[key]['narration_pass_2']['narrations']
    final_json[key] = sort_by_timestamp_frame(final_narr)
  
final_json_storing = final_json.copy()

for key in final_json_storing.keys():
    data = final_json[key]
    if len(data) == 0:
        continue
    else:
        for i in range(len(data) - 1):
            data[i]['stop_timestamp'] = data[i + 1]['timestamp_sec']
            data[i]['stop_frame'] = data[i + 1]['timestamp_frame']

        data[-1]['stop_timestamp'] = data[-1]['timestamp_sec']+3
        data[-1]['stop_frame'] = data[-1]['timestamp_frame']+100

        filtered_data = [{'start_timestamp': item['timestamp_sec'], 'stop_timestamp': item['stop_timestamp'],
                            'start_frame': item['timestamp_frame'], 'stop_frame': item['stop_frame'], 'narration_text': item['narration_text']} for item in data]

        df = pd.DataFrame(filtered_data)
        dataset_outdir = './ego4d_dataset_cooking' # csv 추출해서 저장할 directory 생성
        os.makedirs(dataset_outdir, exist_ok=True)
        df.to_csv(f'{dataset_outdir}/{key}.csv', index=False, encoding='utf-8')