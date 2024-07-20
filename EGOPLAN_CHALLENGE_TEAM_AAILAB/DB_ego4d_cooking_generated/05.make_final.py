import json
import random
import os
import glob
from tqdm import tqdm

def get_all_json_paths(directory_path):
    return glob.glob(os.path.join(directory_path, '*.json'))

def modify_json_for_task(json_paths, output_file_path):
    all_results = []
    count = 0
    for file_path in tqdm(json_paths):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for task in data:
            metadata = task["task_progress_metadata"]
            if len(metadata) >= 3:
                if len(metadata) >= 10:
                    splits = [metadata[:len(metadata)//2], metadata[len(metadata)//2:]]
                else:
                    splits = [metadata]

                for split in splits:
                    if len(split) >= 3:
                        selected_narration = random.choice(split[-3:])
                        answer = selected_narration['narration_text']
                        current_frame = selected_narration['start_frame']
                
                        new_metadata = [n for n in split if n['stop_frame'] <= current_frame]
                        video_id = os.path.basename(file_path).split('.')[0]
                        if video_id.endswith('_output'):
                            video_id = video_id[:-7]
                        
                        result_json = {
                            "sample_id": count+80000,
                            "video_id": video_id,
                            "task_goal": task["task_goal"],
                            "answer": answer,
                            "task_progress_metadata": new_metadata,
                            "negative_answers": list(set([n['narration_text'] for n in metadata if n['narration_text'] != answer])),
                        }
                        all_results.append(result_json)
                        count += 1

    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(all_results, out_file, indent=4, ensure_ascii=False)
    print(f"All modified JSONs saved to {output_file_path}")

directory_path = './convert_narration_result'
output_file_path = './make_final_result/action_database_ego4d_cooking_generated.json'
json_paths = get_all_json_paths(directory_path)

modify_json_for_task(json_paths, output_file_path)
