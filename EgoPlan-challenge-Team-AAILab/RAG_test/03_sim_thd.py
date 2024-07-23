import json

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)
def save_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
filename = './data/test_rag_similarity.json' 
data = load_json(filename)

filtered_data = [item for item in data if item["similarity"] > 0.95]
print(len(filtered_data))
output_filename = './data/test_rag_similarity_95.json'
save_json(output_filename, filtered_data)
print("saved 'test_rag_similarity_95.json'.")
