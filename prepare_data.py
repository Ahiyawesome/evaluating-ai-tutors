import json
import requests

url = "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/main/MRBench/MRBench_V2.json"
output_file = "tutor_eval_data.json"

print(f"Downloading dataset from {url}...")
try:
    raw_data = requests.get(url).json()
    print(f"Successfully loaded {len(raw_data)} conversations.")
except Exception as e:
    print(f"Download failed: {e}")
    exit()

processed_data = []

for item in raw_data:
    history = item.get('conversation_history', '')
    
    tutor_container = item.get('anno_llm_responses', {})
    
    for tutor_name, data in tutor_container.items():
        
        response_text = data.get('response', '')
        
        annotations = data.get('annotation', {})
        label = annotations.get('Mistake_Identification')
        
        if response_text and label:
            entry = {
                "instruction": "Evaluate the following tutor response for Mistake Identification. Output 'Yes', 'No', or 'To some extent'.",
                "input": f"Dialogue:\n{history}\n\nTutor Response:\n{response_text}",
                "output": label
            }
            processed_data.append(entry)

if len(processed_data) > 0:
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    print(f"\nSUCCESS! Created {len(processed_data)} training examples.")
    print(f"Saved to {output_file}")
    
    print("\n--- Example Entry ---")
    print(json.dumps(processed_data[0], indent=2))
else:
    print("Error: Extracted 0 examples. Something is still wrong with the keys.")