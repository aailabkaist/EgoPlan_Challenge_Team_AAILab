import warnings
from transformers import logging
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import json
import re
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

input_file_path = './data/EgoPlan_validation_ego_only.json.json'
output_file_path = './data/target_bert_embedding.json'

with open(input_file_path, 'r') as file:
    data = json.load(file)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def remove_prep_phrase(sentence):
    return re.sub(r',?\s+\w+ the \w+$', '', sentence)

for item in data:
    task_goal = item['task_goal']
    cleaned_choice_a = remove_prep_phrase(item['choice_a'])
    cleaned_choice_b = remove_prep_phrase(item['choice_b'])
    cleaned_choice_c = remove_prep_phrase(item['choice_c'])
    cleaned_choice_d = remove_prep_phrase(item['choice_d'])

    choices = ", ".join([cleaned_choice_a, cleaned_choice_b, cleaned_choice_c, cleaned_choice_d])
    combined_text = f"{choices}"
    print(combined_text)
    embedding = get_bert_embedding(combined_text)
    item['bert_embedding'] = embedding.tolist()  

df = pd.DataFrame(data)
df = df[['sample_id', 'video_id', 'task_goal', 'bert_embedding']]
print(df)

df.to_json(output_file_path, orient='records', lines=True)
