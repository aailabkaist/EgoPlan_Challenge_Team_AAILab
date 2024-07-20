import json
import os
import glob
import pandas as pd
from tqdm import tqdm

def get_all_json_paths(directory_path):
    return glob.glob(os.path.join(directory_path, '*.json'))

def csv_columns_to_string(file_path, columns=['start_frame', 'stop_frame', 'narration_text']):
    df = pd.read_csv(file_path)
    df['start_frame'] = pd.to_numeric(df['start_frame'], errors='coerce')
    df['stop_frame'] = pd.to_numeric(df['stop_frame'], errors='coerce')
    selected_columns_df = df[columns]
    return selected_columns_df

def filter_and_save_narrations(json_paths, csv_directory, output_directory):
    for json_path in tqdm(json_paths):
        with open(json_path, 'r', encoding='utf-8') as original_file:
            content = json.load(original_file)

        if 'clips' not in content:
            continue

        base_name = os.path.basename(json_path)
        file_name_without_extension = os.path.splitext(base_name)[0]
        csv_file_path = os.path.join(csv_directory, f'{file_name_without_extension}.csv')

        if not os.path.exists(csv_file_path):
            continue

        df = csv_columns_to_string(csv_file_path)

        all_narrations = []
        for clip in content['clips']:
            if validate_clip(clip):
                subgoal_narrations = collect_narrations(clip, df)
                if 3 <= len(subgoal_narrations) <= 20:
                    all_narrations.append({
                        "task_goal": clip['subgoal'],
                        "task_progress_metadata": subgoal_narrations
                    })

            if 'sub_clips' in clip:
                for sub_clip in clip['sub_clips']:
                    if validate_clip(sub_clip):
                        subgoal_narrations = collect_narrations(sub_clip, df)
                        if 3 <= len(subgoal_narrations) <= 20:
                            all_narrations.append({
                                "task_goal": sub_clip['secondary_subgoal'],
                                "task_progress_metadata": subgoal_narrations
                            })

        if all_narrations:
            save_narrations(all_narrations, output_directory, file_name_without_extension)

def validate_clip(clip):
    return clip.get('start_timestamp') is not None and clip.get('stop_timestamp') is not None

def collect_narrations(goal, df):
    start_time = int(goal['start_timestamp'])
    stop_time = int(goal['stop_timestamp'])
    narrations = []

    for _, row in df.iterrows():
        if not pd.isna(row['start_frame']) and not pd.isna(row['stop_frame']):
            if row['start_frame'] <= stop_time and row['stop_frame'] >= start_time:
                if int(row['start_frame'])<= int(row['stop_frame']):
                    start_frame = int(row['start_frame'])
                    stop_frame = int(row['stop_frame'])
                    if int(row['start_frame']) == 0:
                        start_frame = 1
                    if int(row['stop_frame']) == 0:
                        stop_frame = 1
                    
                    narrations.append({
                            "narration_text": row['narration_text'],
                            "start_frame": start_frame,
                            "stop_frame": stop_frame
                        })
    return narrations

def save_narrations(all_narrations, output_directory, file_name_without_extension):
    output_file_path = os.path.join(output_directory, f'{file_name_without_extension}_output.json')
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(all_narrations, out_file, indent=4, ensure_ascii=False)

directory_path = './change_json_result'
csv_directory = './ego4d_dataset_cooking' 
output_directory = './make_instance_result'
json_paths = get_all_json_paths(directory_path)

filter_and_save_narrations(json_paths, csv_directory, output_directory)
