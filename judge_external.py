import os
from google import genai
from llama_cpp import Llama
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

LOCAL_JUDGE_PATH = "./llama-3.2-3b-instruct.Q4_K_M.gguf"

USE_OPENAI = False
USE_GEMINI = True  

openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print("Loading Local Judge...")
judge_llm = Llama(
    model_path=LOCAL_JUDGE_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

def get_external_tutor_response(student_message, dialogue_history):

    full_prompt = f"{dialogue_history}\nStudent: {student_message}"
    
    if USE_OPENAI:
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tutor. Help the student but do not reveal the answer directly."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return "GPT-4o", response.choices[0].message.content

    elif USE_GEMINI:
        client = genai.Client(api_key=google_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are a tutor. Context:\n{full_prompt}"
        )
        return "Gemini-Flash", response.text

    else:
        raise Exception("No LLM is used...")

def judge_response(dialogue_context, tutor_response):
    prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an expert pedagogical evaluator. Assess if the AI Tutor correctly identified the student's mistake.

### Criteria:
- **Yes**: Explicitly identifies the error.
- **No**: Misses the error.
- **To some extent**: Vague hint.

Briefly explain how the LLM find or didn't find the mistake.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Dialogue History:
{dialogue_context}

Tutor Response:
{tutor_response}

Did the tutor identify the mistake? (Yes/No/To some extent) Briefly explain how the LLM find or didn't find the mistake.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    output = judge_llm(prompt, max_tokens=512, stop=["<|eot_id|>"], echo=False)
    return output['choices'][0]['text'].strip()


if __name__ == "__main__":

    context = "Tutor: Could you solve this question: what is 100 - 95?"
    
    while True:
        student_input = input("\nStudent (You): ")
        if student_input.lower() == "exit": break
        
        print("asking external tutor...")
        tutor_name, tutor_reply = get_external_tutor_response(student_input, context)
        print(f"\n[{tutor_name} says]: {tutor_reply}")
        
        print(f"\n[Local Judge is evaluating {tutor_name}...] ", end="", flush=True)
        verdict = judge_response(context + f"\nStudent: {student_input}", tutor_reply)
        
        print(f"\n>>> VERDICT (Did the tutor identify the mistake?): {verdict}\n")
        
        context += f"\nStudent: {student_input}\nTutor: {tutor_reply}"