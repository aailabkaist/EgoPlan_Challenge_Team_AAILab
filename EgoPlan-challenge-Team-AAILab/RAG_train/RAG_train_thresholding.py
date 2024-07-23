import json
import os
from tqdm import tqdm

rag_filename = './data/rag/co_occur_objects_10_parsed_obj_noun_EgoPlan_IT_without_answer_train_epiconly.json'
sim_filename = './data/rag/co_occur_objects_10_sim_train_epiconly.json'

with open(rag_filename, 'r') as file:
  rag_data = json.load(file)
print(len(rag_data))
with open(sim_filename, 'r') as file:
  sim_data = json.load(file)
print(len(sim_data))

integrated_target_filename = './data/target_dir/integrated_target_dataset.json'
with open(integrated_target_filename, 'r') as file:
  target_data = json.load(file)
print(len(target_data))
integrated_reference_filename = './data/ref_dir/integrated_ref_dataset.json'
with open(integrated_reference_filename, 'r') as file:
  ref_data = json.load(file)
print(len(ref_data))

def retrieve_augmentation(best_ref_instance):
    if 'answer' in best_ref_instance.keys():
        retrieved_narration = best_ref_instance['task_progress_metadata'].copy()
        new_narr = {}
        new_narr['narration_text'] = best_ref_instance['answer']
        if len(best_ref_instance['task_progress_metadata']) == 0:
            new_narr['start_frame'] = best_ref_instance['current_observation_frame'] + 1
            new_narr['stop_frame'] = best_ref_instance['current_observation_frame'] + 1 + 100
        else:
            new_narr['start_frame'] = best_ref_instance['task_progress_metadata'][-1]['stop_frame'] + 1
            new_narr['stop_frame'] = best_ref_instance['task_progress_metadata'][-1]['stop_frame'] + 1 + 100
        retrieved_narration.append(new_narr)
    else:
        retrieved_narration = best_ref_instance['task_progress_metadata'].copy()
    return retrieved_narration

def add_retrieved_narration_to_target_instance(retrieved_narration):
    if len(retrieved_narration) == 0:
        return ""
    else:
        add_narr = 'In video similar to my situation, a person has done duty along following sequences '
        time_check = 0
        for j in range(len(retrieved_narration)):
            if time_check > retrieved_narration[j]['start_frame']:
                print('warning!!!')
            time_check = retrieved_narration[j]['start_frame']
            if j != len(retrieved_narration) - 1:
                add_narr += f"{j + 1}) {retrieved_narration[j]['narration_text']}, "
            else:
                add_narr += f"{j + 1}) {retrieved_narration[j]['narration_text']}."
        return add_narr

import re
def split_string(s):
    pattern = re.compile(r'^(.*)_(\d+)$')
    match = pattern.match(s)
    if match:
        return str(match.group(1)), int(match.group(2))
    else:
        return None

################# Hyperparamter #################
version = 1                   
threshold = 0.96           # ---> Final similarity threshold
###############################################

add_narr_yes = 0
final_train = []
for key in tqdm(sim_data.keys()):
    top_k = sim_data[key][0]

    target_video_id, target_sample_id = split_string(key)
    ref_video_id, ref_sample_id = split_string(top_k)
    instance = [item for item in target_data if item['sample_id'] == target_sample_id and item['video_id'] == target_video_id][0]
    before_key_nums = len(list(instance.keys()))
  
    if sim_data[key][1] >= threshold:
        ref_taskgoal = [item for item in ref_data if ((item['sample_id'] == ref_sample_id) and (item['video_id'] == ref_video_id))][0]['task_goal']

        ref_candidates = [item for item in ref_data if (item['video_id']==ref_video_id and item['task_goal']==ref_taskgoal)]
        ref_instance_metadata = [len(item['task_progress_metadata']) for item in ref_candidates]
        max_index = ref_instance_metadata.index(max(ref_instance_metadata))
        best_ref_instance = ref_candidates[max_index]
        retrieved_narration = retrieve_augmentation(best_ref_instance)
        
        add_narr = add_retrieved_narration_to_target_instance(retrieved_narration)
        if instance:
            instance['add_narr'] = add_narr
            after_key_nums = len(list(instance.keys()))
    else:
        after_key_nums = len(list(instance.keys()))
        instance['add_narr'] = ""
    
    if before_key_nums != after_key_nums:
        add_narr_yes += 1
    final_train.append(instance)

add_narr_proportion = round(add_narr_yes/len(target_data), 2)
out_dir = './final_train/'
os.makedirs(out_dir, exist_ok=True)
# Filename of final train data (with RAG)
output_json_path = f'final_train_ver{version}_threshold{threshold}_addnarr_prop{add_narr_proportion}.json'
with open(out_dir + output_json_path, 'w') as outfile:
    json.dump(final_train, outfile, indent=4)