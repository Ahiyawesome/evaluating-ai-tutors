import json
import random
from llama_cpp import Llama
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "./llama-3.2-3b-instruct.Q4_K_M.gguf"
DATA_PATH = "tutor_eval_data.json"
TEST_SPLIT = 0.2  


print(f"Loading Judge Model: {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH
    n_gpu_layers=-1, 
    n_ctx=4096,
    verbose=False
)

with open(DATA_PATH, "r") as f:
    full_data = json.load(f)

random.seed(67)
random.shuffle(full_data)
split_idx = int(len(full_data) * (1 - TEST_SPLIT))
test_data = full_data[split_idx:]

print(f"Loaded {len(full_data)} examples. Testing on {len(test_data)} held-out examples.")

y_true = []
y_pred = []

print("\nStarting Evaluation (This may take a while)...")
for i, item in enumerate(test_data):
    prompt_text = f"""<|start_header_id|>system<|end_header_id|>

You are an expert pedagogical evaluator. Assess if the AI Tutor correctly identified the student's mistake.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{item['input']}

Answer with just the label: Yes, No, or To some extent.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    output = llm(
        prompt_text,
        max_tokens=10, # We only need a short label
        stop=["<|eot_id|>", "\n"],
        echo=False
    )
    
    prediction = output['choices'][0]['text'].strip()
    ground_truth = item['output']
    
    prediction_clean = prediction.replace(".", "").lower()
    ground_truth_clean = ground_truth.replace(".", "").lower()
    
    y_pred.append(prediction_clean)
    y_true.append(ground_truth_clean)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_data)}...")

print("\n=== FINAL RESULTS ===")
# We map them back to standard labels for the report if needed, 
# but usually 'yes', 'no', 'to some extent' is fine.
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
print("\nDetailed Report:")
print(classification_report(y_true, y_pred))