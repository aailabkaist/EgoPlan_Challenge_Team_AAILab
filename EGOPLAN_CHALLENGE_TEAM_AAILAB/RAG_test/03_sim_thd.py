import json

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

filename = './data/test_rag_similarity.json' 
data = load_json(filename)

filtered_data = [item for item in data if item["similarity"] > 0.96]

print("saved 'test_rag_similarity.json'.")
