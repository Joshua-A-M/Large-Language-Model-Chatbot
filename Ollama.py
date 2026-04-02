# gpt2_medium_loader.py

# ---- IMPORTS ----
import json
import urllib.request
from gpt_download import download_and_load_gpt2
from C4_implement_GPT_model import GPTModel
from C5_pretraining_unlabeled_data import load_weights_into_gpt

# ---- BASE CONFIG ----
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# ---- MODEL VARIANTS ----
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# ---- FUNCTION TO LOAD MODEL ----
def load_model(CHOOSE_MODEL, sft_path=None):
    """
    Load a GPT-2 model variant and optionally SFT weights.

    CHOOSE_MODEL: str, e.g., "gpt2-medium (355M)"
    sft_path: str or None, path to fine-tuned weights (.pth)
    """
    config = BASE_CONFIG.copy()
    config.update(model_configs[CHOOSE_MODEL])

    # Extract model size for download
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # Download base GPT-2 weights
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    # Initialize model
    model = GPTModel(config)

    # Load fine-tuned SFT weights if provided, else base weights
    if sft_path:
        load_weights_into_gpt(model, sft_path)
    else:
        load_weights_into_gpt(model, params)

    model.eval()  # set to evaluation mode
    return model, config


# ---- FUNCTION TO QUERY OLLAMA-LIKE MODEL ----
def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/chat"
):
    """
    Send a prompt to an Ollama-style local API and return response.
    """
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-medium (355M)"
    SFT_PATH = "gpt2/355M-sft.pth"  # path to your fine-tuned weights

    # Load the model
    model, config = load_model(CHOOSE_MODEL, sft_path=SFT_PATH)
    print(f"Loaded {CHOOSE_MODEL} with config: {config}")

    # Example query (Ollama API)
    test_prompt = "Write a short poem about spring."
    response = query_model(test_prompt)
    print("Ollama response:\n", response)
