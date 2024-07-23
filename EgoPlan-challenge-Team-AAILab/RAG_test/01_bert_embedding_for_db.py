import warnings
from transformers import logging
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

input_file_path = './data/EgoPlan_IT.json'
input_file_path2 = './data/action_data_ego4d_generated_605.json'
output_file_path = './data/db_bert_embedding.json'

with open(input_file_path, 'r') as file:
    train_data = json.load(file)
with open(input_file_path2, 'r') as file:
    gen_data = json.load(file)

train_data = train_data + gen_data 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def remove_prep_phrase(sentence):
    return re.sub(r',?\s+\w+ the \w+$', '', sentence)

def process_item(item):
    task_goal = item['task_goal']
    answer = item['answer']
    cleaned_negative_answers = [remove_prep_phrase(ans) for ans in item['negative_answers']]
    negative_answer = ", ".join(cleaned_negative_answers).replace("'", "")
    choices = ", ".join([answer, negative_answer])
    combined_text = f"{choices}"
    embedding = get_bert_embedding(combined_text)
    item['bert_embedding'] = embedding.tolist()
    return item

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_item, item) for item in train_data]
    processed_data = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        processed_data.append(future.result())

df = pd.DataFrame(processed_data)
df = df[['sample_id', 'video_id', 'task_goal', 'bert_embedding']]

df.to_json(output_file_path, orient='records', lines=True)
