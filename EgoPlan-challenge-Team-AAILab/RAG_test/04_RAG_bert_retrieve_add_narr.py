import json
import re
file_path = './data/test_rag_similarity_95.json'

with open(file_path, 'r') as file:
    rag = json.load(file)

rag = [[item['test_sample_id'], item['most_similar_train_sample_id']] for item in rag]

input_file_path = './data/EgoPlan_IT.json'
input_file_path2 = './data/action_data_ego4d_generated_605.json'

with open(input_file_path, 'r') as file:
    train_data = json.load(file)
with open(input_file_path2, 'r') as file:
    gen_data = json.load(file)

data = train_data + gen_data 
def split_string(s):
    pattern = re.compile(r'^(.*)_(\d+)$')
    match = pattern.match(s)
    if match:
        return match.group(1), match.group(2)
    else:
        return None

add_narr = {}

for sample in rag:
    test_sample_id = sample[0]
    reference_sample_id = sample[1]
    matching_items = [item for item in data if item['sample_id'] == reference_sample_id]
    if len(matching_items) > 0:
        final_ref_instance = matching_items[0]
    else:
        print(f"No matching item found for reference_sample_id: {reference_sample_id}")
        continue 
    final_ref_instance = [item for item in data if item['sample_id'] == reference_sample_id][0]

    new_narr = {}
    if 'answer' in final_ref_instance.keys():
        new_narr['narration_text'] = final_ref_instance['answer']
        if len(final_ref_instance['task_progress_metadata']) == 0:
            new_narr['start_frame'] = final_ref_instance['current_observation_frame']+1
            new_narr['stop_frame'] = final_ref_instance['current_observation_frame']+1+100
        else:
            new_narr['start_frame'] = final_ref_instance['task_progress_metadata'][-1]['stop_frame']+1
            new_narr['stop_frame'] = final_ref_instance['task_progress_metadata'][-1]['stop_frame']+1+100
        new_metadata = final_ref_instance['task_progress_metadata'].copy()
        new_metadata.append(new_narr)
        add_narr[test_sample_id] = new_metadata
    else:
        new_metadata = final_ref_instance['task_progress_metadata'].copy()
        add_narr[test_sample_id] = new_metadata
       

for key in add_narr.keys():
  this_list = add_narr[key]
  value = [item for item in this_list if len(item.keys()) == 3]
  add_narr[key] = value



file_path = './data/EgoPlan_validation_ego_only.json'
with open(file_path, 'r') as file:
    test_data = json.load(file)
new_train = []

other_metadata_keys = add_narr.keys()


for key in other_metadata_keys:
  sample_id = key
  meta_data = add_narr[key]

  now_data = [item for item in test_data if item['sample_id'] == sample_id]
  if len(now_data) == 0:
      continue
  else:
    now_data = now_data[0]
    if len(meta_data) == 0:
        now_data['add_narr'] = None
    else:
        narr = 'In video similar to my situation, a person has done duty along following sequences '
        time_check = 0
        for j in range(len(meta_data)):
          metadata_key = meta_data[j].keys()
          if 'start_frame' not in metadata_key:
            meta_data[j]['start_frame'] = 9999999999
          if time_check >  meta_data[j]['start_frame']:
              print('warning!!!')
          time_check = meta_data[j]['start_frame']
          if j != len(meta_data) - 1:
            if 'narration_text' in meta_data[j].keys():
              narr += f"{j+1}) {meta_data[j]['narration_text']}, "
            else:
              narr += f"{j+1}) {meta_data[j]['narration']}, "
          else:
            if 'narration_text' in meta_data[j].keys():
              narr += f"{j+1}) {meta_data[j]['narration_text']}."
            else:
              narr += f"{j+1}) {meta_data[j]['narration']}."
        now_data['add_narr'] = narr
    new_train.append(now_data)

file_path = './data/test_rag_results_95.json'

with open(file_path, 'w') as file:
    json.dump(new_train, file)

with open(file_path, 'r') as file:
    json_data = json.load(file)
