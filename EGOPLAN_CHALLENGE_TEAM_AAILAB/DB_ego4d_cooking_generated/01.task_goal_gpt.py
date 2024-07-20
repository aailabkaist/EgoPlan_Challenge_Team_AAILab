import base64
import sys
import os
import json
import glob
import argparse
import requests
from tqdm import tqdm
import pandas as pd
import time

TEMPLATE = r""" 
 Your tasks are:
1. Carefully read the provided narrations of this ego-centric video, and connect the narrations by timestamps.
While reading the narrations, please pay attention to inferring the temporal relationships of actions, pre- and post- conditions of actions, object affordance, object states, and possible changes of functional zones that support different activities, etc.
2. Recognize the overall goal of the actions in this ego-centric video.
3. Temporally segment this video into clips. Actions in the same clip complete the same subgoal that is helpful for achieving the overall goal.
4. Further segment each clip into more fine-grained sub-clips. Actions in the same sub-clip are likely to take place in a compact space of the same functional zone without moving around, and focus on a secondary subgoal that is helpful for achieving the corresponding parent-level subgoal.
All the related timestamps in the output should be copied from the given timestamped narrations; the descriptions of the overall goal, subgoals, and secondary subgoals should be detailed with related essential objects; and the output must follow the JSON format:
{"overall_goal": ... , "clips": [{"subgoal": ..., "start_timestamp": ..., "stop_timestamp": ..., "sub_clips": [{“secondary_subgoal": ..., "start_timestamp": ..., "stop_timestamp": ...}, ...]}, ...]}" 
            """


def query_gpt_input(file_name):

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": TEMPLATE + file_name
                    },
                ]
            }
        ],
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()



def save_to_new_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def csv_columns_to_string(file_path, columns=['start_frame', 'stop_frame', 'narration_text']):
    df = pd.read_csv(file_path)
    selected_columns_df = df[columns]
    columns_string = ' '.join(selected_columns_df.astype(str).values.flatten())
    
    return columns_string

def get_all_csv_paths(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    return csv_files


def get_query():
    directory_path = './ego4d_dataset_cooking' 
    csv_paths = get_all_csv_paths(directory_path)
    limit_token = 0

    for i in tqdm(csv_paths):
        csv_string = csv_columns_to_string(i)
        file_name = os.path.basename(i)
        
        try:

            gpt_output = query_gpt_input(csv_string)
            gpt_response = gpt_output["choices"][0]['message']['content']

            if len(gpt_response)<= 1:
                pass
            base_name = os.path.basename(i)
            file_name_without_extension = os.path.splitext(base_name)[0]
            save_to_new_json(gpt_output,output_file_path=f"./gpt_output_cooking/{file_name_without_extension}.json")
            token_num = gpt_output["usage"]['total_tokens']
            limit_token += token_num
            
            if limit_token > 28000:
                print("Waiting for 60 seconds...")
                time.sleep(30)
                limit_token = 0       
        except:
            print(i)




if __name__ == '__main__':

    get_query()
