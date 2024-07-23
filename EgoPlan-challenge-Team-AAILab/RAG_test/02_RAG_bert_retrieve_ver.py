import warnings
from transformers import logging
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Ignore specific warnings
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

# JSON 파일 경로
input_file_path_1 = './data/db_bert_embedding.json'

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

data = load_json_lines(input_file_path_1)

input_file_path = './data/target_bert_embedding.json' #If you want the RAG of test, enter the test data embedding path.
test_data = load_json_lines(input_file_path)

def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def find_most_similar_instance(test_instance, test_data):
    test_embedding = test_instance['bert_embedding']
    max_similarity = -1
    most_similar_instance = None

    for train_instance in test_data:
        train_embedding = train_instance['bert_embedding']
        similarity = calculate_cosine_similarity(test_embedding, train_embedding)
        if float(similarity) >= float(0.9999):
            pass
        else:
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_instance = train_instance

    return {
        'test_sample_id': test_instance['sample_id'],
        'most_similar_train_sample_id': most_similar_instance['sample_id'],
        'similarity': max_similarity
    }

def process_instance(test_instance):
    return find_most_similar_instance(test_instance, data)

if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_instance, test_data), total=len(test_data)))

    output_file_path = './data/test_rag_similarity.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(results, output_file, indent=4)
