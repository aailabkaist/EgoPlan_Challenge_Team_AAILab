
# RAG_train

  

## Dataset

  

+ Target: Egoplan_IT.json

+ Reference: Egoplan_IT.json / action_data_ego4d_generated.json

  

## Running

  
"""
bash run.sh
"""
  

## Process

1. RAG_train.py

+ Create the *Integrated target dataset* and *Integrated reference dataset*

	+  *Integrated target dataset*: Egoplan_IT.json

	+  *Integrated reference dataset*: Egoplan_IT.json + action_data_ego4d_generated.json

+ For each of *Integrated target dataset* / *Integrated reference dataset*, “task goal” is parsed into objects and verbs for each instance and saved.

+ For each instance of *Integrated target dataset*, get the most similar instance from *Integrated reference dataset* based on the cosine similarity between the parsed “task goal”.

2. RAG_train_thresholding.py

+ To filter out potentially irrelevant retrieved action sequences, only RAG results for instances with cosine similarity above threshold are retained and utilized as the final train dataset.