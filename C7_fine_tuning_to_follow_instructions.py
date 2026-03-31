#  P A G E  204
#   f I N E - T U N I N G  T O  F O L L O W  I N S T R U C T I O N S
#   implement the process for fine-tuning an LLM to follow human instructions
#   instruction fine-tuning is one of the main techniques behind developing LLMs for chatbot applications,
#   person assistants, and other conversational tasks
#   two main ways for fine-tuning an LLM: classification and fine-tuning on LLM to follow instructions

#   7.1 Introduction to instruction fine-tuning
#   pretraining an LLM involves a training procedure where it learns to generate one word at a time
#   resulting pretrained LLM is capable of text completion: it can finish sentences or write
#   text paragraphs given a fragment as input
#   pretrained LLMs often struggle with specific instructions, such as "Fix
#   the grammar in this text" or "Convert this text into passive voice."
#   supervised instruction fine-tuning
#   focus on improving the LLM's ability to follow such instructions and generate a desired response
#   7.2 Preparing a dataset for supervised instruction fine-tuning
#   download and format the instruction dataset for instruction fine-tuning a pretrained LLM
#   dataset consists of 1,100 instruction-response pairs
#   dataset was created for the tutorial
#   alternative publicly available instruction datasets in appendix B
import json
import os
import urllib


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
#   the output is 1100
print("Number of entries:", len(data))

#   print one of the entries
print("Example entry:\n", data[50])

#   take a look at another example
print("Another example entry:\n", data[999])

#   instruction fine-tuning involves training a model on a dataset where the input-output pairs are
#   explicitly provided
#   various methods to format these entries for LLMs
#   two different formats:
#   prompt styles used in the training of notables LLMs such as Alpaca and Phi-3
#   define a format_input function that can be used to convert the entries in the data list into
#   the Alpaca-style input format
def format_input(entry):
    instruction_text = (
        f"\n\nBelow is an instruction that describes a task. "
        f"\n\nwrite a response that appropriately completes the request. "
        f"\n\n### Instruction: \n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input: \n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

#   lets test it to dataset entry data[50]
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

#   the format_input skips the optional ### input: section if the 'input' field is empty
#   can test out by applying the format_input function to entry dataset[999]
model_input = format_input(data[999])
desired_response = f"\n\n### Response: \n{data[999]['output']}"
print(model_input + desired_response)

#   divide the dataset into training, validation, and test sets analogous to what we have done with the spam
#   classification dataset in the previous chapter
#   Use 85% of the data for training
train_portion = int(len(data) * 0.8)
#   use 10% for testing
test_portion = int(len(data) * 0.1)
#   use remaining 5% for validation
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length", len(test_data))

#   ready for the core implementation of the instruction fine-tuning process
#   focus on developing the method for constructing the training batches for fine-tuning
#   the LLM

#   7.3 Organizing data into training batches
#   next step focuses on constructing the training batches effectively
#   define a method that will ensure our model receives the formatted training data during the fine-tuning process
#   batching process for instruction fine-tuning requires us to create our own custom collate function that is plugged into the DataLoader
#   a collate function is responsible for taking a list of individual data samples and merging them into a single
#   batch that can be processed efficiently by the model during training
#   steps of batching process
#   1.  format data using prompt template   2.  Tokenize formatted data     3.  Adjust to the same length with padding tokens   4.  Replace padding tokens with placeholders


#   step 1
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


#   step 2
#   want to accelerate training by collecting multiple training exmaples in a batch
#   which necessitates padding all inputs to a similar length
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

#   step 3
#   custom collate function pads the training examples in each batch to the same length while allowing different batches to
#   have different lengths
def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    #   Finds the longest sequence in the batch
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst = []

    #   Pads and prepares inputs
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        #   Removes extra padded token added earlier
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    #   Converts the list of inputs to a tensor and transfers it to the target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


#   custom_collate_draft_1 is designed to be integrated into a PyTorch DataLoader
#   it can also function as a standalone tool
#   use it here independently to test and verify that it operates as intended:

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
#   output shows all inputs have been padded to the length of the longest input list, inputs_1,
#   containing five token IDs
print(custom_collate_draft_1(batch))

#   need to create batches with the target token IDs corresponding to the batch of input IDs
#   these target IDs are crucial because they represent what we want the model to generate and what we
#   need during training to calculate the loss for the weight updates
#   modify our custom collate function to return the target token IDs in addition to the input token IDs

#   the target token IDs match the input token IDs but are shifted one position to the right
#   this allows the LLM to learn how to predict the next token in a sequence

#   the following collate function generates the target token IDs from the input token IDs
def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst,targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
        (batch_max_length - len(new_item))
        )
        #   Truncates the last token inputs
        inputs = torch.tensor(padded[:-1])
        #   Shifts +  to the right for targets
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

#   next, assign a -100 placeholder value to all padding tokens
#   This special value allows us to exclude these padding tokens from contributing to the training loss calculation, ensuring
#   that only meaningful data influences model learning
#   retain one end-of-text token, ID 50256, in the target list
#   retaining it allows the LLM to learn when to generate an end-of-text token in response or instructions
#   which we use as an indicator that the generated response is complete

#   modify the custom collate function to replace token with ID 50256 with -100 in the target lists
#   introduce an allowed_max_length parameter to optionally limit the length of the samples

def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        #   Pads sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        #   Truncates the last token for inputs
        inputs = torch.tensor(padded[:-1])
        #   shifts + 1 to the right for target
        targets = torch.tensor(padded[1:])

        #   Replaces all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        #   Optionally truncates to the maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

#   try the collate function on the sample batch
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

#   demonstration purposes
#   here's how we might calculate the cross entropy loss during training when the model predicts a
#   sequence of tokens
logits_1 = torch.tensor(
    #   predictions for 1st token
    [[-1.0, 1.0],
    #  predictions for 2nd token
    [-0.5, 1.5]]
)

#   Correct token indices to generate
targets_1 = torch.tensor([0, 1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

#   adding an additional token ID affects the loss calculation:

logits_2 = torch.tensor(
    #   predictions for 1st token
    [[-1.0, 1.0],
    #  predictions for 2nd token
    [-0.5, 1.5],
     #  New third token ID prediction
     [-0.5, 1.5]]
)

targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

#  what happens if we replace the third target token ID with -100:

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
#   the loss on these three training examples is identical to the loss calculated from the two training examples earlier
#   cross entropy loss function ignores the third entry in the targets_3 vector, the token ID corresponding to -100
#   the default setting of the cross entropy function in PyTorch is cross_entropy(..., ignore_token_index=100)
#   it ignores targets labeled with -100
#   want to keep one 50256 (end-of-text) token Id in the targets because it helps the LLM
#   to learn to generate end-of-text tokens which we can use as an indicator that a
#   response is complete
print("loss_1 == loss_3:", loss_1 == loss_3)

#   it is also common to mask out the target token IDs that correspond to the instruction
#   by masking out the LLM's target token IDs corresponding to the instruction, the cross
#   entropy loss is only compared for the generated response target IDs
#   the model is trained to focus on generating accurate responses rather than memorizing
#   instructions, which can help reduce overfitting

#   Page 223
#   7.4 Creating data loaders for an instruction dataset
#   implement both InstructionDataset objects and the custom_collate_fn function into PyTorch data loaders
#   loaders will automatically shuffle and organize the batches for the LLM instruction fine-tuning process
#   moved the data onto the target device in the main training loop
#   this part of the collate function offers the advantage of performing this device transfer process as a
#   background process outside the training loop
#   preventing it from blocking the GPU during model training
#   following code initializes the device variable:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   Uncomment the two lines ot use the GUP on an Apple Silicon chip
#   if torch.backends.mps.is_available():
#   device = torch.device("mpa")"
#   this will either print "Device: cpu" or "Device: cuda", depending on your machine
print("Device:", device)

#   next to reuse the chosen device setting in custom_collate_fn when we plug it into the PyTorch
#   DataLoader class, we use the partial function from Python's functools standard library to create a
#   version of the function with the evice argument prefilled
#   set the allowed_max_length to 1024, which truncates the data to the maximum context length supported
#   by the GPT-2 model

from functools import partial
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

#   next, set up the data loaders and use the custom collate function for the batching process:
from torch.utils.data import DataLoader

#   You can try to increase this number if parallel Python processes are supported by your operating system
num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

#   examine the dimensions of the input of the input and target batches generated by the training loader

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

#   Page 226
#   7.5 Loading a pretrained LLM
#   load a pretrained GPT model  that we want to fine-tune
#   instead of using the smallest 124-million-parameter model, load the medium-sized model with 355 million parameters
#   124-million-parameter model is too limited in capacity to achieve satisfactory results via instruction fine-tuning
#   smaller models lack the necessary capacity to learn and retain the intricate patterns and nuanced behaviors required
#   for high-quality instruction-following tasks
#   loading the pretrained models requires the same code as when we pretrained the data and fine-tuned it for
#   classification

from gpt_download import download_and_load_gpt2
from C4_implement_GPT_model import GPTModel
from C5_pretraining_unlabeled_data import load_weights_into_gpt

BASE_CONFIG = {
    #   Vocabulary size
    "vocab_size": 50257,
    #   Context length
    "context_length": 1024,
    #   Dropout rate
    "drop_rate": 0.0,
    #   Query-key-value bias
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

#   assess the pretrained LLM's performance on one of the validation tasks by comparing its output to the expected response.
#   this will give us a baseline understand of how well the model performs on an instruction-following
#   task right out of hte box, prior to fine-tuning
#   will help to appreciate the effect of fine-tuning later on

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

#   next, generate the model's response using the same generate function used to pretrain the model in chapter 5:
from C5_pretraining_unlabeled_data import generate, text_to_token_ids, token_ids_to_text

#   generate function returns the combined input and output text.
#   when evaluating the model's performance on  a specific task, we often
#   want to focus solely on the model's generated response
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)

generated_text = token_ids_to_text(token_ids, tokenizer)

#   to isolate the model's response text, we need to subtract the length of the input instruction from the start of the generated_text:
response_text = generated_text[len(input_text):].strip()
print(response_text)

#   Page 229
#   7.6 Fine-tuning the LLM on instruction data
#   take the loaded pretrained model and train it using the prepared instruction dataset
#   for the fine-tuning process, reuse the loss calculation and training functions implemented in chapter 5

from C5_pretraining_unlabeled_data import (
calc_loss_loader,
train_model_simple
)

#   calculate the initial loss for the training and validation sets

model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )

#   returns 3.825908660888672
print("Training loss:", train_loss)
#   returns 3.7619335651397705
print("Validation loss:", val_loss)

#   can now proceed to train the model
#   set up the training process including initializing the optimizer, setting the number of epochs, and defining the
#   evaluation frequency and starting context to evaluate generated LLM response during
#   training based on the first validation set instruction

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
#   the output will display the training progress over two epochs, where a steady
#   decrease in losses indicates improving ability to follow instructions and
#   generate appropriate response
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

#   examine the training and validation loss curves to gain additional insights into the model's learning process
#   use the same plot_losses function

from C5_pretraining_unlabeled_data import plot_losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#   model's performance on both the training and validation sets improves substantially
#   over the course of training
#   rapid decrease in losses during the initial phase indicates that the
#   model quickly learn meaningful patterns and representation from the data
#   as training progresses to the second epoch, the losses continue to decrease but at a
#   slower rate, suggesting that the model is fine-tuning its learned
#   representation and converging it to a stable solution
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

#   Page 233
#   7.7 Extracting and saving responses
#   evaluate the LLM's performance on the held-out test set
#   first extract the model-generated responses for each input in the test dataset
#   and collect them for manual analysis
#   then evaluate the LLM to quantify the quality of the responses

#   to complete the response instruction step, use the generate function
#   print the model responses alongside the expected test set answers for
#   the first three test set entries
#   presenting them side by side for comparison

torch.manual_seed(123)

#   Iterates over the first three test set samples
for entry in test_data[:3]:
    input_text = format_input(entry)
    #   Uses the generate function
    #   returns the combined input and output text
    #   use the slicing and the .replace() on the generated_text contents to extract the model's response
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size =BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("--------------------------------------")

#   use the generate method in the same manner as before
#   iterate over the entire text_set
#   instead of printing the model responses, add them to the test_set dictionary

from tqdm import tqdm
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text =  (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    #   indent for pretty-printing
    json.dump(test_data, file, indent=4)

#   verify that the responses have been correctly added to the teest_set dictionary
#   by examining one of the entries
print(test_data[0])

#   save the model as gpt2-medium355M-sft.pth file to be able to reuse it in
#   future projects
import re

#   Removes white spaces and parentheses from file name
file_name = f"{re.sub(r'[()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

#   Page 238
#   7.8 Evaluating the fine-tuned LLM
#   toe evaluate test set responses in an automated fashion, utilize an existing instruction-fine-tuned
#   8-billion parameter Llama 8 model
import psutil

#   Ensure that Ollama is still running
#   evaluate the responses generated by our fine-tuned model that prompts the Llama 3 model to
#   rate our fine-tuned model's responses on a scale
#   from 0 to 100 based on the given test set responses as reference
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
)

print("Ollama running:", check_if_running("ollama"))

#   an alternative to the ollama run command for interacting with the model is through its REST
#   API using Python.
import urllib.request
# import requests
def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11435/api/chat"
):
    data = {
        #   Creates the data payload as a dictionary
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        #   settings for deterministic responses
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    #   Converts the dictionary as a JSON-formatted string and encodes it to bytes
    payload = json.dumps(data).encode("utf-8")

    #   Creates a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""
    #   Sends the request and captures the response
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    # Send the POST request
    # with requests.post(url, json=data, stream=True, timeout=30) as r:
    #     r.raise_for_status()
    #     response_data = ""
    #     for line in r.iter_lines(decode_unicode=True):
    #         if not line:
    #             continue
    #         response_json = json.loads(line)
    #         if "message" in response_json:
    #             response_data += response_json["message"]["content"]
    #
    return response_data

model = "llama3"
result = query_model("What do Llama eat?", model)
print(result)


#   first, apply the approach to the first three examples from the test set

#   generated responses show that the Llama 3 model provides reasonable evaluations and is
#   capable of assigning partial points when a model's answer is not entirely correct
for entry in test_data[:3]:
    prompt = (
    f"Given the input `{format_input(entry)}` "
    f"and correct output `{entry['output']}`, "
    f"score the model response `{entry['model_response']}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response.")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")


#   modify the prompt to just generate integer scores ranging from 0 to 100, where
#   100 represents the best possible score
#   allows us to calculate an average of its performance

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            #   modified instruction line to only return the score
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

#   apply the generate_model_scores function to the entire test_data set
scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores; {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")