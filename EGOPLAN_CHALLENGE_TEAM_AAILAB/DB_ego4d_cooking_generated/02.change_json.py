import json
import os
import glob
from tqdm import tqdm

def get_all_json_paths(directory_path):
    return glob.glob(os.path.join(directory_path, '*.json'))

def save_to_new_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def clean_json_content(content):
    content = content.replace("'", '"')
    return content

def extract_json_from_markdown(content):
    start = content.find('```json') + 7  
    end = content.find('```', start)  
    content = content[start:end].strip()  
    return content

def extract_content_to_new_json():
    directory_path = './gpt_output_cooking'
    json_paths = get_all_json_paths(directory_path)
    
    for original_file_path in tqdm(json_paths):
        try:
            with open(original_file_path, 'r', encoding='utf-8') as original_file:
                content = original_file.read()
            content = clean_json_content(content)
            
            data = json.loads(content)
            json_content = data['choices'][0]['message']['content']
            json_content = extract_json_from_markdown(json_content)
            
            content_dict = json.loads(json_content)
            
            base_name = os.path.basename(original_file_path)
            file_name_without_extension = os.path.splitext(base_name)[0]
            new_file_path = f"./change_json_result/{file_name_without_extension}.json"
            save_to_new_json(content_dict, new_file_path)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error in {original_file_path}: {e}")
        except Exception as e:
            print(f"Error processing {original_file_path}: {e}")

extract_content_to_new_json()
