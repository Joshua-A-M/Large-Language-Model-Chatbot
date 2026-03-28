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



