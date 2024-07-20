import os
import json
import glob

def find_and_process_json_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_paths = glob.glob(os.path.join(input_directory, '*.json'))
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for task in data:
            task['task_goal'] = convert_to_verb_object(task['task_goal'])
            for item in task['task_progress_metadata']:
                item['narration_text'] = convert_to_verb_object(item['narration_text'])
        
        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(output_directory, base_name)
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            json.dump(data, new_file, ensure_ascii=False, indent=4)

def convert_to_verb_object(text):
    parts = text.split()
    filtered_parts = [part.replace('#', '') for part in parts if part.lower() not in ['#c', 'c', '#o', 'c', '#c']]
    result_text = ' '.join(filtered_parts)
    if result_text.lower().startswith('and '):
        result_text = result_text[4:]
    
    return result_text.capitalize()
input_directory = './make_instance_result'
output_directory = './convert_narration_result'
find_and_process_json_files(input_directory, output_directory)
