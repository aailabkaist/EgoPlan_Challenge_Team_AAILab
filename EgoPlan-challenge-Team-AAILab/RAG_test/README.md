
# RAG_test

  

## Dataset

  

+ Target: EgoPlan_validation.json or EgoPlan_test.json

+ Reference: Egoplan_IT.json and action_data_ego4d_generated_605.json

  

## Running

  
```
bash run.sh
```
  

## Process

1. 01_bert_embedding_for_db.py

+ Input: This script extracts embeddings from the action database where the type of query is answers. It loads `EgoPlan_IT.json` and `action_data_ego4d_generated_605.json` from the action database and extracts BERT embeddings based on answers.

+ Output: Outputs the embeddings to `db_bert_embedding.json`.


2. 01_bert_embeding_for_test_RAG.py
+ input: This script requires a dataset that needs testing. Depending on the requirement, set the dataset to either `EgoPlan_validation.json` or `EgoPlan_test.json`. The script will extract the BERT embedding for queries that have an answer.

+ Output: The output will be saved as `target_bert_embedding.json`.

3. 02_RAG_bert_retrieve_ver.py
+ input: `db_bert_embedding.json` and `target_bert_embedding.json`
+ output: `test_rag_similarity.json`

4. 03_sim_thd.py
+ input: `test_rag_similarity.json`
+ output: 'test_rag_similarity_95.json`
+ This script allows you to input a desired threshold value. Only the RAG results with similarity scores exceeding this threshold will be retained.

5. 04_RAG_bert_retrieve_add_narr.py
+ input: `test_rag_similarity_95.json`, `EgoPlan_IT.json` and `action_data_ego4d_generated_605.json`
+ output: `test_rag_results_95.json`

6. 05_RAG_bert_retrieve_blank_fill.py
+ input: `test_rag_results_95.json` and `EgoPlan_validation.json` or `EgoPlan_test.json`
+ output: `final_test_rag_bert_embedding_ver_95.json`


The final JSON file for Test RAG is `final_test_rag_bert_embedding_ver_95.json`.


