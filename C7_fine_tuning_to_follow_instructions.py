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
        f"Below is an instruction that describes a task. "
        f"write a response that appropriately completes the request. "
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
#   want to acceleate training by collecting multiple training exmaples in a batch
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

#   the target token IDs match the input toekn IDs but are shifted onp osition to the right
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
#   introduce an allowed_max_length parameter to ptionally limit the length of the samples

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
#   here's how we moght calculate the cross entropy losss during training when the model predicts a
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