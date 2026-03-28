#   P A G E  1 6 9
#   C H A P T E R  S I X
#   F I N E - T U N I N G  F O R  C L A S S I F I C A T I O N
#   have coded the LLM architecture, pretrained it, and learned how to import pretrained weights from an external source
#   such as OpenAI, into the model
#   Now we will fine-tune the LLM on a specific target task, such as classifying text
#   classify text messages as "spam" or "not spam"
#   This is step 8

#   6.1 Different categories of fine-tuning
#   most common ways to fine-tune language models are instruction fine-tuning and classification fine-tuning
#   instruction fine-tuning involves training a language model on a set of tasks using specific instructions to improve its ability to understand and execute
#   tasks described in natural language prompts
#   in classification fine-tuning, the model is trained to recognize a specific stet of class labels, such as "spam" and "not-spam"
#   restricted to predicting classes it has encountered during its training
#   an instruction fine-tuned model can undertake a broader range of tasks

#   6.2 Preparing the dataset
#   modify and classification fine-tune the GPT model by fist downloading and preparing the dataset
#   to provide an intuitive and useful example of classification fine-tuning,
#   will work with a text message dataset that consists of spam and not-spam messages

#   first step is to download the dataset
import requests
import urllib.request
import zipfile
import os
from pathlib import Path


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except (requests.exceptions.RequestException, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

#   after executing the code, the dataset is saved as a tab-separated text file, SMSSpamCollection.tsv
#   in the sms_spam_collection folder.
#   load it into a pandas DataFrame

import pandas as pd
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
#   Renders the data frame in a Jupyter notebook. Alternatively, use print(df)
df
print(df)

#   because we prefer a small dataset which will facilitate faster fine tuning of the LLM, we choose
#   to undersample the dataset to include 747 instances from each class

#   use this code to undersample and create a balanced dataset
def create_balanced_dataset(df):
    #   Counts the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        #   randomly samples "ham" instances to match the number of "spam" instances
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        #   combines ham subset with "spam"
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

#   next, convert the "string" class labels "ham" and "spam" into integer class labels 0 and 1
#   process is similar to converting text into token IDs
#   Instead of using the GPT vocabulary, which consists of more than 50,000 words,
#   we are dealing with just two token IDs:0 and 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

#   next, create a random_split function to split the dataset into three parts: 70%
#   for training, 10% for validation, and 20% for testing
#   These ratios are common in machine learning to train, adjust, and evaluate models

def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    #   Shuffles the entire DataFrame
    train_end = int(len(df) * train_frac)
    #   Calculates split indices
    validation_end = train_end + int(len(df) * validation_frac)

    #   Splits the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(
    #   Test size is implied to be 0.2 as the remainder
    balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

#   6.3 Creating data loaders
#   Will develop PyTorch data loaders conceptually similar to those implemented while working with text data
#   utilized a sliding window technique to generate uniformly sized text chunks, which we then grouped into batches for more
#   efficient model training
#   Each chunk functioned as an individual training instance
#   are now working with a spam dataset that contains text messages of varying lengths
#   to batch these messages as we did with the text chunks, have two primary options:
#   1.  truncate all messages to the length of the shortest message in the dataset or batch
#   2.  pad all messages to the length of the longest message in the dataset or batch
#   first option is computationally cheaper, but it may result in significant information loss if shorter
#   messages are much smaller than the average or longest messages potentially reducing model performance
#   opt for the second option which preserves the entire content of all messages
#   to implement batching, where all messages are padded to the length of the longest message in the dataset
#   we add padding tokens to all shorter messages
#   we use "<|endoftext|>" as a padding token
#   can add the token ID corresponding to "<|endoftext|>" to the encoded text messages
#   can double-check whether the token ID is correct by encoding the "<|endoftext|>" using the GPT-2 tokenizer from the tiktoken package
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

#   need to implement a PyTorch Dataset, which specifies how the data is loaded
#   and processed before we can instantiate the data loaders
#   we define the SpamDataset class, which implements the concepts.
#   This SpamDataset class handles several key tasks: it encodes the text messages into
#   token sequences, identifies the longest sequence in the training dataset
#   ensures that all other sequences are padded with a padding token to match the length of the longest sequence

import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        #   pre-tokenizes texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            #   Truncates sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        #   Pads sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

#   the SpamDataset class loads data from the CS files created earlier, tokenizer the text using the GPT-2
#   tokenizer from tiktoken, and allows us to pad or truncate
#   the sequences to a uniform length determined by either the longest sequence or a
#   predefined maximum length

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

#   longest sequence length is stored in the dataset's max_length attribute
#   prints the number of tokens in the longest sequence
print(train_dataset.max_length)

#   next, pad the validation and test sets to match the length of the longest training sequence.
#   any validation and test set samples exceeding the length of the longest training example are truncated using encoded_text[:self.max_length]
#   in the SpamDataset code
#   This truncation is optional
#   can set max_length=None for both validation and test sets

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

#   using the dataset as inputs, can instantiate the data loaders similarly to when we were working
#   with text data
#   the targets represent class labels rather than the next tokens in the text

#   code creates the training, validation, and test set data loaders that load the text messages and labels in batches of size 8
from torch.utils.data import DataLoader

#   This setting ensures compatibility with most computers
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

#   ensure that the data loaders are working and are, indeed, returning batches of the
#   expected size, we iterate over the training loader and then print the tensor dimensions
#   of the last batch:

for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

#   input batches consist of eight training examples with 120 tokens each
#   the label tensor stores the class labels corresponding to the eight training examples
#   lastly, print the total number of batches in each dataset:
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

#   prepare the model for fine-tuning
#   6.4 Initializing a model with pretrained weights
#   start by initializing pretrained model
#   employ the same configurations used to pretrain unlabeled data:

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

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
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "N-heads": 25},
    "gpt2-x1 (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

#   import the download_and_load_gpt2 function from the gpt_download.py file and reuse
#   the GPTModel class and load_weights_into_gpt function fom pretraining to load the downloaded
#   weights into the GPT model

from gpt_download import download_and_load_gpt2
from C4_implement_GPT_model import GPTModel
from C5_pretraining_unlabeled_data import load_weights_into_gpt
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

#   after loading the model weights into the GPTModel,reuse the text generation utility function to ensure that
#   the model generates coherent text

from C5_pretraining_unlabeled_data import generate_text_simple, text_to_token_ids, token_ids_to_text
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
#   should show the model generates coherent text which is indicative that the model weights have been loaded correctly
print(token_ids_to_text(token_ids, tokenizer))

#   see whether the model already classifies spam messages by prompting it with instructions:

text_2 = (
    "Is the following text 'smap'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.' "
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

#   model is struggling to follow instructions
print(token_ids_to_text(token_ids, tokenizer))

#   6.5 Adding a classification head
#   must modify the pretrained LLM to prepare it for classification fine-tuning
#   replace the original output layer, which maps the hidden representation to a vocabulary of 50,257,
#   with a smaller output layer that maps to two classes: 0 and 1
#   first print the model architecture via print(model)
#   output neatly lays out the architecture
#   the GPTModel consists of embedding layers followed by 12 identical transformer blocks,
#   followed by a final LayerNorm  and the output layer, out_head
print(model)

#   next, replace the out_head with a new output layer that we will fine-tune
#   first freeze the model or "make all layers nontrainable"

for param in model.parameters():
    param.requires_grad = False

#   then replace the output layer(model.out_head) which originally maps the layer
#   inputs to 50, 257 dimensions, the size of the vocabulary

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

#   use BASE_CONFIG["emb_dim"] which is equal to 768 in the "gpt2-small (124M)" model to keep the code
#   more general
#   can also use the same code to work with the larger GPT-2 model variants
#   new model.out_head output layer has its requires_grad attribute set to true by default,
#   which means it's the only layer in the model that will be updated during
#   training
#   to make the final LayerNorm and last transformer block trainable, set their
#   respective requires_grad to True:

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

#   can feed the model an example text identical to the previously used example
#   text

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
#   shape:(batch_size, num_tokens)
print("Inputs dimensions:", inputs.shape)

#   then, pass the encoded token IDs to the model as usual:

with torch.no_grad():
    outputs = model(inputs)
#   interested in fine-tuning this model to return a class label indicating whether
#   a model input is "spam" or "not spam."
#   don't need to fine-tune all four output row
#   we can focus on a single output token
#   focus on the last row corresponding to the last output token
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

#   to extract the last output token from the output tensor, use the following code:
print("Last output token:", outputs[:, -1, :])

#   still need to convert the values into a class-label prediction
#   now ready to transform the last token into class label predictions and calculate
#   the model's initial prediction accuracy.

#   Page 190
#   6.6 Calculating the classification loss and accuracy
#   must implement the model evaluation function used during fine-tuning
#   work with 2-dimensional instead of 50,257 dimensional outputs
#   example:
#   values of the tensor corresponding to the last tokens are tensor([[-3.5983, 3.9902]])
print("Last output token:", outputs[:, -1, :])

#   obtain the class label
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
#   the code returns 1, meaning the model predicts that the input text is "spam"
print("Class label:", label.item())

#   can simplify the code without using softmax:
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

#   concept can be used to compute the classification accuracy, which measures the percentage of correct predictions
#   across a dataset
#   to determine the classification accuracy, we apply the argmax-based prediction code to all
#   examples in the dataset and calculate the proportion of correct predictions
#   by defining a calc_accuracy_loader function

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                #   logits of last output token
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
            (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples

#   use the function to determine the classification accuracies across various datasets estimated
#   from 10 batches for efficiency

device = torch.device("cuda" if torch.cuda.is_available() else "cup")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)

val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)

test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

#   the prediction accuracies are near a random prediction, which would be 50% in this case
#   to improve the prediction accuracies, need to fine-tune the model
#   must first define the loss function we will optimize during training
#   objective is to maximize the spam classification accuracy of the model which means
#   that the preceding code should output the correct class labels: 0 and 1
#   because classification accuracy is not a differentiable function, use cross-entropy loss as
#   a proxy t maximize accuracy

#   calc_loss_batch function remains the same but focus on optimizing only the last token, model(input_batch)[:, -1, :], rather
#   than all tokens, model(input_batch)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    #   logits of last output token
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

#   use the calc_loss_batch function to compute the loss for a single batch obtained from the
#   previously defined data loaders
#   to calculate the loss for all batches in a data loader, define the calc_loss_loader function
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        #   Ensures the number of batches doesn't exceed batches in data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

#   similar to calculating the training accuracy, now compute the initial loss for each data set:

#   Disables gradient tracking for efficiency because we are not training yet
with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

#   next, implement a training function to fine-tune the model, which means adjusting the model to minimize
#   the training set loss
#   this helps increase the classification accuracy
#   Page 195
#   6.7 Fine-tuning the model on supervised data

#   must define and use the training function to fine-tune the pretrained LLM and improve its
#   spam classification accuracy
#   the training loop is the same overall training loop used for pretraining
#   calculate hte classification accuracy instead of generating a sample text to evaluate the model

#   track the number of training examples seen instead of the number of tokens
#   calculate the accuracy after each epoch instead of printing a sample text
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    #   Initializes lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    #   Main training loop
    for epoch in range(num_epochs):
        #   Sets model to training mode
        model.eval()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            #   Resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            #   Calculates loss gradients
            loss.backward()
            #   Updates model weights using loss gradients
            optimizer.step()
            #   New: tracks examples instead of tokens
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        #   Calculates accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

#   evaluate function is identical to the one we used for pretraining

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

#   next initialize the optimizer, set the number of training epochs, and initiate the
#   training using the train_classifier_simple function
import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

#   We then use Matplotlib to plot the loss function for the training and validation set
import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):

    fig, ax1 = plt.subplots(figsize=(5, 3))

    #   Plots training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    #   Creates a second x-axis for examples seen
    ax2 = ax1.twiny()
    #   Invisible plot for aligning ticks
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    #   Adjust layout to make room
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

#   there is little to no indication of overfitting
#   there is no noticeable gap between the training and validation set losses
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

#   using the same plot_values function, let's now plot the classification accuracies

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

#   the model achieves a relatively high training and validation accuracy after epochs 4 and 5

plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label="accuracy"
)

#   must calculate the performance metrics for the training, validation, and tests sets across
#   the entire dataset
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

#   the training and test set performances are almost identical
#   minimal overfitting of the training data
#   validation set accuracy is typically higher than the test set accuracy because the model development
#   often involves tuning hyperparameters to perform well on the validation set
#   which might not generalize as effectively to the test set

#   Page 200
#   6.8 Using the LLM as a spam classifier
#   ready to classify spam messages
#   use the fine-tuned GPT-based system classification model
#   the following classify_review function allows data preprocessing steps similar to those used in the SpamDataset
#   after processing text into token IDs, function uses the model to predict an integer class label
def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
text_2 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award "
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

torch.save(model.state_dict(), "review_classifier.pth")

# model_state_dict = torch.load("review_classifier.pth, map_location=device")
# model.load_state_dict(model_state_dict)
