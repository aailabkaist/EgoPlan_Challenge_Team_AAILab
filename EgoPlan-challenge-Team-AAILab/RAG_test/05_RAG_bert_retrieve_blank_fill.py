import json

test_blank_file = './data/test_rag_results_95.json'
test_file = './data/EgoPlan_validation_ego_only.json'

# Load JSON data from files
with open(test_blank_file, 'r') as f:
    test_blank = json.load(f)

with open(test_file, 'r') as f:
    test_data = json.load(f)



existing_sample_ids = {entry['sample_id'] for entry in test_blank}
for entry in test_data:
    sample_id = entry['sample_id']
    if sample_id not in existing_sample_ids:
        entry['add_narr'] = ""
        
        test_blank.append(entry)

test_blank.sort(key=lambda x: x['sample_id'])

output_file = "./data/final_test_rag_bert_embedding_ver_95.json"
with open(output_file, 'w') as f:
    json.dump(test_blank, f, indent=4)

print(f"Filled and sorted test_blank saved to {output_file}")
