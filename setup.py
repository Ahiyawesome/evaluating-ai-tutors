import os
from huggingface_hub import hf_hub_download

REPO_ID = "Ahiyawesome/llama-3.2-3b-custom" 
FILENAME = "llama-3.2-3b-instruct.Q4_K_M.gguf"

if not os.path.exists(FILENAME):
    print(f"Model not found. Downloading {FILENAME} from Hugging Face...")
    
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print("Download complete!")
else:
    print("Model already exists. Ready to run.")