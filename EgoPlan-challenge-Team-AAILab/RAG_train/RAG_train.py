import csv
import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import re
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

np.seterr(invalid='ignore')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def egoPlan_parse_obj_verb_without_answer(target_json_list, narration=False, negative=False, test=False):
    parsed_dict = {}

    directory = './data/'

    narration_mark = ''
    negative_mark = ''

    if narration:
        narration_mark = '_narr'

    if negative:
        negative_mark = '_neg'

    for target_json in tqdm(target_json_list, desc="Parsing JSON files"):

        narration_mark += '_' + target_json[:-5]

        # JSON 파일 로드
        with open(directory + target_json, 'r') as file:
            data = json.load(file)

        for item in data:
            video_id = item['video_id']
            tasks = [item['task_goal']]
            if narration:
                try:
                    tasks.extend([narration['narration_text'] for narration in item['task_progress_metadata']])
                except:
                    pass
            if negative:
                try:
                    tasks.extend(item['negative_answers'])
                except:
                    pass

            item_key = video_id + '_' + str(item['sample_id'])
            if item_key not in parsed_dict:
                parsed_dict[item_key] = {'verbs': [], 'objects': []}

            for task in tasks:
                tokens = word_tokenize(task)
                tagged_tokens = pos_tag(tokens)

                for word, tag in tagged_tokens:
                    if tag.startswith('NN'):  # 명사
                        if word not in parsed_dict[item_key]['objects']:
                            parsed_dict[item_key]['objects'].append(word)
                    elif tag.startswith('VB'):  # 동사
                        if word not in parsed_dict[item_key]['verbs']:
                            parsed_dict[item_key]['verbs'].append(word)

    if test:
        with open(f'{directory}parsed_obj_noun{narration_mark}{negative_mark}_without_answer.json', 'w') as out_file:
            json.dump(parsed_dict, out_file, indent=4)
    else:
        with open(f'{directory}parsed_obj_noun{narration_mark}{negative_mark}_without_answer.json', 'w') as out_file:
            json.dump(parsed_dict, out_file, indent=4)

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def cosine_similarity(vec1, vec2):
    """ Calculate cosine similarity between two vectors """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def precompute_co_occurrence_embeddings(data, criteria):
    vectorizer = CountVectorizer()
    documents = [' '.join(content[criteria]) if isinstance(content[criteria], list) else content[criteria]
                 for _, content in data.items()]

    X = vectorizer.fit_transform(documents)
    embeddings = X.toarray()
    return embeddings, vectorizer

def save_co_occurrence_embeddings(embeddings, vectorizer, embeddings_filename, vectorizer_filename):
    with open(embeddings_filename, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(vectorizer_filename, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_co_occurrence_embeddings(embeddings_filename, vectorizer_filename):
    with open(embeddings_filename, 'rb') as f:
        embeddings = pickle.load(f)
    with open(vectorizer_filename, 'rb') as f:
        vectorizer = pickle.load(f)
    return embeddings, vectorizer

def co_occurrence_sim(query_content, embeddings, vectorizer, data_keys, k):

    query_list = query_content['verbs'] + query_content['objects']
    query_vec = vectorizer.transform([' '.join(query_list)]).toarray()[0]

    similarities = {}
    for idx, vec in enumerate(embeddings):
        sim_score = cosine_similarity(query_vec, vec)
        similarities[data_keys[idx]] = sim_score

    sorted_videos = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_videos = sorted_videos[:k]
    return top_k_videos

def split_string(s):
    pattern = re.compile(r'^(.*)_(\d+)$')
    match = pattern.match(s)
    if match:
        return str(match.group(1)), int(match.group(2))
    else:
        return None

def process_videos(parsing_target_file_name, parsing_reference_file_name, metric='co_occur', criteria='task_goal', k=3, test=False):
    directory = './data/'

    with open(directory + parsing_target_file_name, 'r') as file:
        data = json.load(file)
    data_keys = list(data.keys())

    with open(directory + parsing_reference_file_name, 'r') as file:
        ref_data = json.load(file)
    ref_keys = list(ref_data.keys())

    embeddings_filename = f'{directory}co_occurrence_embeddings_{parsing_target_file_name[:-5]}.pkl'
    vectorizer_filename = f'vectorizer_{parsing_target_file_name[:-5]}.pkl'
    if not os.path.exists(embeddings_filename) or not os.path.exists(vectorizer_filename):
        embeddings, vectorizer = precompute_co_occurrence_embeddings(data, criteria)
        save_co_occurrence_embeddings(embeddings, vectorizer, embeddings_filename, vectorizer_filename)
    else:
        embeddings, vectorizer = load_co_occurrence_embeddings(embeddings_filename, vectorizer_filename)

    embeddings_filename = f'{directory}co_occurrence_embeddings_{parsing_reference_file_name[:-5]}.pkl'
    vectorizer_filename = f'vectorizer_{parsing_reference_file_name[:-5]}.pkl'
    if not os.path.exists(embeddings_filename) or not os.path.exists(vectorizer_filename):
        ref_embeddings, ref_vectorizer = precompute_co_occurrence_embeddings(ref_data, criteria)
        save_co_occurrence_embeddings(embeddings, vectorizer, embeddings_filename, vectorizer_filename)
    else:
        ref_embeddings, ref_vectorizer = load_co_occurrence_embeddings(embeddings_filename, vectorizer_filename)

    with open(directory + 'target_dir/integrated_target_dataset.json', 'r') as file:
        target_data = json.load(file)
    with open(directory + 'ref_dir/integrated_ref_dataset.json', 'r') as file:
        ref_data = json.load(file)

    final_train = []
    sim_info = {}
    for video_id, content in tqdm(data.items(), desc="Processing videos"):
        top_k = co_occurrence_sim(content, ref_embeddings, ref_vectorizer, ref_keys, k)
        top_k = [item for item in top_k if item[0] != video_id][0]
        sim_info[video_id] = top_k

        target_video_id, target_sample_id = split_string(video_id)
        ref_video_id, ref_sample_id = split_string(top_k[0])

        # best instance의 metadata retrieve
        best_ref_instance = [item for item in ref_data if ((item['sample_id'] == ref_sample_id) and (item['video_id'] == ref_video_id))][0]
        retrieved_narration = retrieve_augmentation(best_ref_instance)

        add_narr = add_retrieved_narration_to_target_instance(retrieved_narration)
        instance = next((item for item in target_data if item['sample_id'] == target_sample_id and item['video_id'] == target_video_id), None)
        if instance:
            instance['add_narr'] = add_narr
        if "negative_answers" not in instance.keys():
            golden_choice_idx = instance.get('golden_choice_idx')
            all_choices = {k: v for k, v in instance.items() if k.startswith('choice_')}
            negative_answers = [v for k, v in all_choices.items() if k[-1] != golden_choice_idx]
            instance['negative_answers'] = negative_answers
            for key in all_choices.keys():
                del instance[key]
            if 'golden_choice_idx' in instance:
                del instance['golden_choice_idx']
        final_train.append(instance)

    # 결과 파일 저장
    output_json_path = f'{metric}_{criteria}_{k}_{parsing_target_file_name[:-5]}_train_epiconly.json'
    sim_json_path = f'{metric}_{criteria}_{k}_sim_train_epiconly.json'
    out_dir = './data/rag/'
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + output_json_path, 'w') as outfile:
        json.dump(final_train, outfile, indent=4)
    with open(out_dir + sim_json_path, 'w') as outfile:
        json.dump(sim_info, outfile, indent=4)

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

def save_embeddings(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def make_integrated_target_dataset(target_json_list, from_dir, target_dir):
    target_dataset = []
    for filename in tqdm(target_json_list, desc="Integrating target datasets"):
        with open(from_dir + filename, 'r') as file:
            data = json.load(file)
        target_dataset.extend(data)
    with open(target_dir + 'integrated_target_dataset.json', 'w') as outfile:
        json.dump(target_dataset, outfile, indent=4)

def make_integrated_ref_dataset(target_json_list, from_dir, ref_dir):
    ref_dataset = []
    for filename in tqdm(target_json_list, desc="Integrating reference datasets"):
        with open(from_dir + filename, 'r') as file:
            data = json.load(file)
        ref_dataset.extend(data)
    with open(ref_dir + 'integrated_ref_dataset.json', 'w') as outfile:
        json.dump(ref_dataset, outfile, indent=4)

def main():
    from_dir = './data/'
    target_json_list = ['EgoPlan_IT.json']
    reference_json_list = ['EgoPlan_IT.json', 'action_data_ego4d_generated.json']

    target_dir = './data/target_dir/'
    ref_dir = './data/ref_dir/'
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    make_integrated_target_dataset(target_json_list, from_dir, target_dir)
    make_integrated_ref_dataset(reference_json_list, from_dir, ref_dir)

    egoPlan_parse_obj_verb_without_answer(target_json_list, False, False)
    egoPlan_parse_obj_verb_without_answer(reference_json_list, False, False)

    target_text = ('_'.join(target_json_list)).replace('.json', '')
    parsing_target_file_name = f'parsed_obj_noun_{target_text}_without_answer.json'
    ref_text = ('_'.join(reference_json_list)).replace('.json', '')
    parsing_reference_file_name = f'parsed_obj_noun_{ref_text}_without_answer.json'
    process_videos(parsing_target_file_name, parsing_reference_file_name,
                   metric='co_occur', criteria='objects', k=10)


if __name__ == "__main__":
    main()
