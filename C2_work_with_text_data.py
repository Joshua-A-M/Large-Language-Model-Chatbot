#   C H A P T E R  T W O
#   W O R K I N G  W I T H  T E X T  D A T A
import urllib.request
#   First, we will discuss how we split input text into individual tokens, a required preprocessing step for creating embeddings for an LLM
#   These tokens are either individual words or special characters
# The text we will tokenize for LLM training is "The Verdict," a short story by Edith Wharton
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

#   load the-verdict.txt file using Python's standard file reading utilities
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))

print(raw_text[:99])

#   Goal is to tokenize this 20,479-character short story into individual words and special characters that we can turn into embeddings for LLM training
#   How can we split this text to obtain a list of tokens? Will use regular expressions for this first example
#   Capitalization helps LLMs distinguish between proper nouns and common names, understand sentence structure, and learn to generate text with proper capitalization
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
#   This list still includes whitespace characters
print(result)

#   Let's modify the regular expression splits on whitespaces (\a), commas, and periods ([,.])
result = re.split(r'([,.]\s)', text)
print(result)

#   Let's remove whitespace
result = [item for item in result if item.strip()]
print(result)

"""   Modify the tokenization scheme so that it can also handle other types of punctuation, 
   such as question marks, quotation marks, and the double-dashes along with special additional characters
"""

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

#   Apply the  tokenization scheme  to Edith Wharton's entire short story
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#   Should print 4690, the number of tokens in this text (without whitespaces)
print(len(preprocessed))
#   Print the first 30 tokens for a quick visual check
print(preprocessed[:30])

#   Converting tokens into token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

#   page 25
#   create the vocabulary and print its first 51 entries (for illustration purposes)
#   Te dictionary will contain individual tokens associated with unique intege labels
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

#   Want to convert outputs of LLM from numbers back into text requires a way to turn token IDs into text
#   create an inverse version of the vocabulary that maps token IDs back to the corresponding text tokens
#   implement a complete tokenizer class in Python with an anecode method that splits tokens and carries
#   out the string-to-integer mapping
#   Will implement a decode method that carries out the reverse integer-to-string mapping
class SimpleTokenizerV1:
    """
    SimpleTokenizerV1 instantiates new tokenizer objects via an existing vocabulary
    Store Vocabulary as a class attribute for access in the encode and decode methods
    encode function Processes input text into token id
    decode function converts token IDs back into text
    Use regular expression to remove spaces before the specified punctuation
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        #   adjust the tokenizer to handle unknown tokens
        # preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        #   Replace spaces before the specified punctuations
        # text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    """
    SimpleTokenizerV1 instantiates new tokenizer objects via an existing vocabulary
    Store Vocabulary as a class attribute for access in the encode and decode methods
    encode function Processes input text into token id
    decode function converts token IDs back into text
    Use regular expression to remove spaces before the specified punctuation
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        #   adjust the tokenizer to handle unknown tokens
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])


        #   Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

#   instantiate a new tokenizer object from the SimpleTokenizerV1 class
#   Tokenize a passage from Edith Wharton's short story
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

#   turn the token IDs back into text using the decode method
print(tokenizer.decode(ids))

#   New text sample
#   This will result in KeyError: "Hello"
#   `It is not contained in the vocabulary
#   Highlights the need to consider large and diverse training sets to extend the vocabulary when working on LLMs
text = "Hello, do you like tea?"

#   Can modify the toeknizer to use an <|unk|> token if it encounters word that is not part of the vocabulary
#   Add a token between unrelated texts
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

#   New Vocabulary size is 1,132
#   Print the last five entries of the updated vocabulary
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

#   The ne tokenizer replaces unknown words with <|unk|> tokens
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|> ".join((text1, text2))
print(text)

#   tokenize the sample text using the SimpleTokenizerV2 on the vocab
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

#   detokenize the text for a quick sanity check
print(tokenizer.decode(tokenizer.encode(text)))

#   B Y T E  P A I R  E N C O D I N G
#   BPE tokenizer was used to train LLMs such as GPT-2, GPT-3, and the original model used in ChatGPT
#   open source library called tiktoken

from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

#   Instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|encoder|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

#   convert the token IDs back into text using the decode method
strings = tokenizer.decode(integers)
print(strings)

#   Data Sampling with a sliding Window
#   Page 35

#   This is the next step in creating the embeddings for the LLM: generate the input-target pairs required for training an LLM
#   LLMs are pretrained by predicting the next word in a text
#   Implement a data loader that fetches the input-target pairs from the training dataset usinga sliding window

#   Tokenize the whole "The Verdict" short story with the BPE tokenizer
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
#   This will return 5145, the total number of tokens in the training set after applying the BPE tokenizer

#   Remove the first 50 tokens from the dataset for demonstration purposes as it results in a slightly more interesting text passage
enc_sample = enc_text[50:]

#   one of the easiest and most intuitive ways to create the input-target pairs for the next-word prediction task is to create two variables, x and y
#   x contains the input tokens and y contains the targets

#   The context size determines how many tokens are included in the input
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x: {x}")
print(f"y: {y}")

#   By processing the inputs along with the targets, whic are the inputs shifted by one position, we can create the next-word prediction tasks
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #   Everything left of the arrow refers to the input an LLM would receive
    #   the token ID on the right side of the arrow represents the target token ID that the LLM is supposed to predict
    print(context, "---->", desired)

#   convert the token IDs into text
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#   One more task before we canturn the tokens into embeddings: implementing an efficient data loader that iterates over the input dataset and
#   returns the inputs and targets as PyTorch tensors (multidimensional arrays)
#   two tensors: an input tensor containing the text that the LLM sees and a target tensor that includes the targets for the LLM to predict

import torch
from torch.utils.data import Dataset, DataLoader
#   defines how individual rows are fetched from the dataset
#   each row consists of a number of token IDs assigned to an input_chunk_tensor
class GPTDatasetV1(Dataset):
    """
    Efficient data loader implementation: use the PyTorch's built-in Dataset classes
    Tokenize the entire text
    Return the total number of rows in the dataset
    Return a single row from the dataset
    Use a sliding window to chunk the book into overlapping sequences of max_length
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

#   use the GPTDatasetV1 to load the inputs in batches via a PyTorch Dataloader
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True,
                         drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        #   drop_last=True drops the last batch if it is shoter than the specified batch_size to prevent loss spikes during training
        drop_last=drop_last,
        #   number of CPU processes to use for preprocessing
        num_workers=num_workers
    )

    return dataloader

#   Test the dataloader with a batch size of 1 for an LLM with a context size of 4
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch= next(data_iter)
#   The first_batch variable contains two tensors: the first store the input token IDs,the second tensor store the target token IDS
#   It is common to train LLMs with input sizes of at least 256
print(first_batch)

#   Understand meaning of stride=1
second_batch = next(data_iter)
print(second_batch)

#   can use data loader to sample with a batch size greater than one

dataloader = create_dataloader_v1(
    raw_text,
    #   Increasing batch_size to 5 utilizes the data set: we don't skip a single word
    batch_size=8
    , max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

#   C R E A T I N G  T O K E N  E M B E D D I N G S
#   Last step in preparing the input text for LLM training is to convert the token IDs into embedding vectors
#   Initialize these embedding weights with random values
#   serves as the starting point for the LLM's learning process
#   A continuous vector representation, or embedding, is necessary since GPT-like LLMs are deep neural networks trained with the backpropagation algorithm

#   token ID to embedding vector conversion example
input_ids = torch.tensor([2, 3, 5, 1])

#   sample vocabulary and dimension
vocab_size = 6
output_dim = 3

#   Instantiate an embedding layer in PyTorch, setting rhe random seed to 123 for reproducibility purposes
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#   weight matrix of embedding layer contains small, random values that are optimized during LLM training as part of the LLM optimization itself
#   Has six rows and three columns
#   one column for each of the three embedding dimensions
print(embedding_layer)

#   apply it to as token ID
#   should return tensor([[-0.4015, 0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>]
print(embedding_layer(torch.tensor([3])))

#   Apply single token ID conversion into a three-dimensional embedding vector to all four input IDs
print(embedding_layer(input_ids))

#   E N C O D I N G  W O R D  P O S I T I O N S
#   shortcoming of token embeddings with LLMs is that their self attention mechanism doesn't have a notion of position
#   or order for the otkens without a sequence
#   deterministic, position-independent embeddingof the token ID is good for reproducibility purposes.
#   Helpful to inject additional position information into the LLM
#   use two broad categories of position-aware embeddings: relative positional embeddings and absolute positional embeddings
#   absolute positional embeddings are directly associated with specific positions in a sequence
#   relative positional embedding isrelated to the relative position or distance between tokens

#   consider more realistic and sueful embedding sizes and encode the inpute tokes into 256-dimensional vector representation

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#   instantiate the date loader
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape: \n", inputs.shape)

#   `use the embedding layer to embed these token IDs into 256-dimensional vectors
#   each token ID is now embedded as a 256-dimensional vector
token_embeddings = token_embedding_layer(inputs)
#   returns torch.size([8, 4, 256])
print(token_embeddings.shape)

#   For a GPT model's absolute embedding approach, we just need to create another embedding layer that has the same embedding dimension as the token_embedding_layer

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#   input is usually a placeholder vector torch.arange(context_length)
#   context_length is a variable that represents the supported input size of the LLM
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#   returns torch.Size([4, 256])
print(pos_embeddings.shape)

#   add the positional tensors to the token embeddings
input_embeddings = token_embeddings + pos_embeddings
#   output is torch.Size([8, 4, 256])
print(input_embeddings.shape)




