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










#   C H A P T E R  2
#   C O D I N G  A T T E N T I O N  M E C H A N I S M S

#   first step of implementing self-attention is to compute the intermediate values w, referred to as attention scores
inputs = torch.tensor(
    [
    [0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]
     ]
)

#   how to calculate the intermediate attention scores between the query token and each input token
#   compute the dot product of the query, x ^(2) with every other input token:
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
    #   the computed attention scores are tensor([0.9544, 1.4950, 0.8434, 0.7070, 1.0865])
    print(attn_scores_2)

#   Next step is to normalize each of the attention scores to obtain attention eights that sum up to 1
#   useful for interpretation and maintaining training stability in an LLM

attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_scores_2_tmp)
#   The attention weights will sum up to 1
print("Sum:", attn_scores_2_tmp.sum())

#   more common and advisable to use the softmax function for normalization
#   better at managing extreme values and offers more favorable gradient properties during training.
#   softmax function ensures that the attention weights are always positive
#   makes the output interpretable as probabilities or relative importance
#   may encounter numerical instability problems such as overflow and underflow when dealing with large or small input values
#   advisable to use the PyTorch implementation of softmax which has been extensively optimized for performance
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#   Next step: calculate the context vector z^(2) by multiplying the embedded input tokens, x^(i) with the corresponding attention weights
#   and then summing the resulting vectors
#   context vector z^(2) is the weighted sum of all input vectors, obtained by multiplying each input vector by its corresponding attention weight

#   The second input token is the query
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
#   The results are tensor([0.4419, 0.6515, 0.5683])
print(context_vec_2)

#   Page 61
#   Next, generalize this procedure for computing context vectors to calculate all context vectors simultaneously

#   modify code to compute all context vectors instead of only the second one, z^(2)
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
#   can use for loops or matrix multiplication. For loops are slower
attn_scores = inputs @ inputs.T
#   each element in the tensor represents an attention score between each pair of inputs

print(attn_scores)

#   Normalize ea;h row so tha the values in each row sum to 1

#   In the context of using PyTorch, the dim parameter in functions like torch.softmax specfies the dimension of the input tensor
#   along which the function will be computed
#   By setting dim=-1, instruct the softmax function to apply the normalization along the last dimension of the attn_scores tensor
attn_weights = torch.softmax(attn_scores, dim=-1)


print(attn_weights)

#   verify that the rows indeed all sum to 1

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

#   Third step
#   use attention weights to compute all context vectors via matrix multiplication
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

#   double-check that the code is correct b comparing the second row with the context vector x^(2)
print("Previous 2nd context vector:", context_vec_2)

#   Page 64
#   Next step is to add trainable weights, enabling the LLM to learn from data and improve its performance on specific tasks
#   implement the self-attention mechnaism used in the original transformer architecture, the GPT models, and most other popular LLMs
#   scaled dot-product attention

#   compute context vectors as weighted sums over the input vectors specific to a certain input element
#   introduction of weight matrices that are updated ruing model training
#   crucial so that the model can learn to produce "good" context vectors

#   Computing the attention weights tep by step
#   implement the self attention mechanism step by step by introducing the three trainable weight matrices: Wq, Wk, and Wv
#   Used to project the embedded input tokens x ^(i), into query, key, and value vectors

#   start by computing only one context vector z ^(2) then modify the code to calculate all context vectors

#   The second input element
x_2 = inputs[1]
#   The input embedding size
d_in = inputs.shape[1]
#   The output embedding size
d_out = 2

#   In GPT like models, the input and output dimensions are usually the same

#   Initialize the three weight matrices

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#   requires_grad=False reduces clutter in the outputs, but if we were to use the weight matrices for model training, we would set required_grad=True
#   to update these matrices during model training

#   next compute the query, key, and value vectors:
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
#   returns a two-dimensional vector since we set the number of columns of the corresponding weight matrix via d_out, to 2

print(query_2)

#   require the key and value vectors for all input elements because the yare involved in computing the attention weights with respect to the query q^(2)
keys = inputs @ W_key
values = inputs @ W_value
#   should successfully project the six input tokens from a three-dimensional onto a two-dimensional embedding space
print("keys.shape:", keys.shape)
print("values.shape", values.shape)

#   Next, compute the attention scores:
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
#   returns tensor(1.8524)
print(attn_score_22)

#   Can generalize this computation to all attention scores via matrix multiplication:

#   all attention scores for given query
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

#   move from attention scores to the attention weights
#   compute the attention weights by scaling the attention scores and using the softmax function
#   now we scale the attention scores by dividing them by the square root of the embedding dimension of the keys

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
#   resulting attention weights are tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.1820])
print(attn_weights_2)

#   final step is to compute the context vectors
#   compute the context vector as a weighted sum over the value vectors
#   attention weights serves as a weighting factor that weighs the respective importance of each value vector
#   can use the matrix multiplication to obtain the output in one step

context_vec_2 = attn_weights_2 @ values
#   results are tensor([0.3061, 0.8210])
print(context_vec_2)

#   Page 70
#   Next, generalize the code to compute all context vectors in the input sequence, z ^(1) to z^(T)
#   Helpful to organize this code into a Python class

import torch.nn as nn
class SelfAttention_v1(nn.Module):
    """
    class derived from nn.Module which is a fundamental building block of PyTorch models that provides necessary functionalities
    for model layer creation and management

    __init__ method initializes trainable weight matrices for queries, keys, and values
    each weight transforms the input dimension d_int to an output dimension d_out

    During the forward pass, using the forward method, compute the attention scores by multiplying queries and keys, normalizing these scores using softmax
    create a context vector by weighting the values with these normalized attention scores
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

#   can improve the SelfAttention_v1 class by utilizing PyTorch's nn.Linear layers, which performs matrix multiplication when the bias units are disabled.
#   nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
#   result differs from SelfAttention_v1 because they use different initial weights for the weight matrix
print(sa_v2(inputs))

#   want the self-attention mechanism to consider only the tokens that appear prior to the current position when predicting the next token in a sequence
#   Casual attention or mashed attention is a specialized form of self attention that restricts amodel to only consider previous and current inputs in a sequence
#   when processing any given token when computing attention scores
#   for each token processed, we mask out the future tokens, which come after the current otken in the input text
#   mask out the attention weights above the diagonal and normalize the nonmasked attention weights such that the attention weights sum to 1 in each row.

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

#   implement the second step using PyTorch's tril function to create a mask where the values above the diagonal are zero:
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

#   multiply the mask with the attention weights to zero-out the values above the diagonal
masked_simple = attn_weights*mask_simple
print(masked_simple)

#   third step is to renormalize the attention weights to sum up to 1 again in each row.
#   dividing each element in each row by the sum in each row
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

#   implement more efficient masking by creating a mask with 1s above the diagonal and then replacing these 1s with negative infinity (-inf) values:
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

#   apply the softmax function to these masked results
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

#   another useful tweak for reducing overfitting when training the LLMs
#   3.5.2 Masking additional attention weights with dropout
#   dropout is a technique where randomly selected hidden layer units are ignored during training, effectively "dropping" them out.
#   helps prevent overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units.
#  dropout is only  used during training and is disabled afterward
#   In transformer architecture, including models like GPT, dropout in the attention mechanism is typically applied at two specific times: after calculating the attention weights
#   or after applying the attention weights to the value vectors
#   apply the dropout mask after computing the attention eights
torch.manual_seed(123)
#   dropout rate of 50% = masking out half of the attention weights
#   The values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 = 2
#   scaling is crucial to maintain the overall balance of the attention weights
#   ensures that the average influence of the attention mechanism remains consistent during both the training and inference phases
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))

#   Implementing a compact casual attention class
#   class will serve as a template for developing multi-head attention - the final attention class we will implement
#   ensure that the code can handle batches consisting of more than one input so that CasualAttention class supports the batch
#   outputs produced by the data loader

#   simulate such batch inputs:
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

#   CausalAttention class is similar to the SelfAttention class ecept the droput and casual mask components are added
class CausalAttention(nn.Module):
    """
    class derived from nn.Module which is a fundamental building block of PyTorch models that provides necessary functionalities
    for model layer creation and management

    __init__ method initializes trainable weight matrices for queries, keys, and values
    each weight transforms the input dimension d_int to an output dimension d_out

    During the forward pass, using the forward method, compute the attention scores by multiplying queries and keys, normalizing these scores using softmax
    create a context vector by weighting the values with these normalized attention scores
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        #   Added a dropout layer
        #   Register buffer call
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1),
        )

    def forward(self, x):
        #   transpose dimensions 1 and 2, keeping the batc dimension at the first position
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        #   In PyTorch, operations with a trailing underscore are performed in-place avoiding unnecessary memory copies
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
#   resulting context vector: context_vecs.shape torch.Size([2, 6, 2
print("context-vecs.shape:", context_vecs.shape)

#   Extending single-head attention to multi-head attention
#   This is the final step: extend the casual attention class over multiple heads (multi-head attention)
#   "multi-head" refers to dividing the attention mechanism into multiple "heads," each operating independently
#   a single casual attention module can be considered single-head attention, where there is only one set of attention weights processing
#   the input sequentially

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

#   if we use this MultiHeadAttentionWrapper class with two attention heads (via num_heads = 2) and CausalAttention output dimensions
#   d_out = 2, we get a four-dimensional context vector
#   use the MultiHeadAttentionWrapper class similar to the CausalAttention class as an example
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

#   Have implemented a MultiHeadAttentionWrapper that combined multiple single-head attention modules.
#   Improve the forward method from processing the single-head attention modules sequentially
#   process the heads in parallel by computing the outputs for all attention heads simultaneously via matrix multiplication

#   Page 86
#   Implementing multi-head attention with weight splits
#   combine MultiheadAttentionWrapper and CausalAttention into a single MultiHeadAttention class.

class MultiHeadAttention(nn.Module):
    """
    itnegrates the multi-head functionality within a single class
    splits the input into multiple heads by reshaping the projected query, key, and value tensors and then
    combines the results from these heads after computing attention
    """

    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        #   Reduces the projection dim to match the desired output dim
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #   Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        #   Tensor shape: (b, num_tokens, d_out)

        # implicitly split the matrix by adding a num_heads dimension
        #   Then unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        #   Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #   computes dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        #   Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # uses the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #   Tensor shape: (b, num_tokens, n_heads, head_dim
        context_vec = (attn_weights @ values).transpose(1, 2)

        #   Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        #   Adds an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec


#   The MultiHeadAttention class can be used similar to the SelfAttention and CausalAttention classes:
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)










#   C H A P T E R  F O U R
#   I M P L E M E N T I N G  A  G P T  M O D E L  F R O M  S C R A T C H  T O  T E X T
#   Code the other building blocks of an LLM and assemble them into a GPT-like model
#   LLMs such as GPT (Generative pretrained transformer) are large deep neural network architectures designed to generate new text one word at a time

#   implement the core structure of the GPT model, including its transformer blocks
#   used smaller embedding dimensions for simplicity, ensuring that the concepts and examples could comfortably
#   fit on a single page
#   scaling up to the size of a small GPT-2 model
#   specify the configuration of the small GPT-2 model via a python dictionary:
GPT_CONFIG_124M = {
    "vocab_size": 50257,    #   Vocabulary size
    "context_length": 1024,     #   Context length
    "emb_dim": 768,     #   Embedding dimension
    "n_heads": 12,      #   Number of attention heads
    "n_layers": 12,     #   Number of layers
    "drop_rate": 0.1,    #   Dropout rate
    "qkv_bias": False   #   Query-Key-Value bias
}

#   use concise variable names for clarity and to present long lines of code:
#   vocab size refers to a vocabulary of 50,257 words use by the BPE tokenizer
#   context_length denotes the maximum number of input tokens the model can handle via the positional embeddings
#   emb_dim represents the embedding size, transforming each token into a 768-dimensional vector
#   n_heads indicates the count of attention heads in the multi-head attention mechanism
#   n_layers specifies the number of transformer blocks in the model
#   drop_rate indicates the intensity of the dropout mechanism to prevent overfitting
#   qkv_bias determines whether to include a bias vector in the Linear layers of the multi-head attention for query, key, and value computations

#   implement a GPT placeholder architecture
import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    """
    defines simplified version of a GPT-like model using PyTorch's neural network module
    architecture consists of token and positional embeddings, dropout, a series of transformers blocks, a
    final layer normalization, and a linear output layer

    Configuration is passed in via a Python dictionary
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #   Uses a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        #   Uses a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    #   describes the data flow through the model:
    #   it computes token and positional embeddings for the input indices, applies dropout, processes
    #   the data through the transformer blocks, applies normalization, produces logits with the linear output layer
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    """
    A simple placeholder class that will be replaced by a real TransformerBlock later
    """
    def __init__(self, cfg):
        super().__init__()

    #   This block does nothing and ust returns its input
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    """
    A dimple placeholder class that will be replaced by a real LayerNorm later
    """

    #   The parameters here are just to mimic the LayerNorm Interface
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

#   initialize a new 124-million-parameter DummyGPTModel instance and feed it the tokenized batch:

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

#   Page 99
#   Normalizing activations with layer normalization
#   Training deep neural networks with many layers can sometimes prove challenging due to problems like vanishing or exploding gradients
#   lead to unstable training dynamics
#   make it difficult for the network to effectively adjust its weights
#   leaning process struggles to find a set of parameters for the neural network that minimizes the loss function

#   implement layer normalization to improve the stability and efficiency of neural network training
#   adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1 (unit variance)
#   speeds up the convergence to effective weights and ensures consistent, reliable training.
#   In GPT-2 and modern transformer architectures, layer normalization is typically applied before and after the multi-head attention module
#   and before the final output layer

# implement a neural network layer with five inputs and six outputs that we apply to two input examples (example)
#   neural network layer consists of a Linear layer followed by a non-linear activation function, ReLU (rectified linear unit), which is
#   a standard activation function in neural networks
#   it simply thresholds negative inputs to 0, ensuring that a layer outputs only positive values
#   explains why the resulting layer output does not contain any negative values
torch.manual_seed(123)
#   creates two training examples with the dimensions (features) each
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

#   before applying layer normalization to the outputs, examine the mean and variance:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance: \n", var)

#   apply layer normalization to the layer outputs we obtained earlier.
#   operation consists of subtracting the mean and dividing by the square root of the variance (standard deviation):

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs: \n", out_norm)
print("Mean:\n", mean)
print("Variance: \n", var)

#   to improve readability, we can also turn off the scientific notation when printing tensor values by setting sci_mode to False:

torch.set_printoptions(sci_mode=False)
print("Men:\n", mean)
print("variance:\n", var)

#encapsulate this process ina PyTorch module that we can us in the GPT model later
class LayerNorm(nn.Module):
    """
    operates on the last dimension of the input tensor x, which represents the embedding dimension
    (emb_dim).
    The variable eps is a small constant (epsilon) added to the variance to prevent division by
    zero during normalization
    Scale and shift are two trainable parameters that the LLM automatically adjust during training if it is determined
    that doing so would improve the model's performance on its training task.
    Allows the model to learn appropriate scaling and shifting that best suit the data it is processing
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


#   try the LayerNorm module in practice and apply it to the batch input
#   Page 105
#   4.3 Implementing a feed forward network with GELU activations
#   implement a small neural network submodule used as aprt of the transformer block in LLMs
#   Implement the GELU activation function
#   ReLU activation function is normally used in deep learning but several other activation functions are employed beyond ReLU
#   including GELU Gaussian error inear unit and SwiGLU Swish-gated linear unit

#   more complex and smooth activation functions incorporating Gaussian and sigmoid-gated linear units
#   GELU activation function can be implemented in several ways. Here is the exact version:
#   GELU(x)=x⋅Φ(x) cumulative distribution function of the standard
#   Gaussian distribution.
#   Common to implement a computationally cheaper approximation: GELU(x) 0 0.5 - x * (1 + tanh [ sqrt of (2 / x) * (x + 0.044713 * x cubed)]
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

#   let's plot these functions side by side GELU and ReLU

import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)

plt.tight_layout()
plt.show()

#   The smoothness of GELU can lead to better optimization properties during training
#   it allows for more nuanced adjustments to the model's parameters
#   GELU allows for a small non-zero output for negative values unlike ReLU
#   ReLU has a sharp corner at zero which can make optimization harder in networks that are very deep or have complex architectures

# use the GELU function to implement the small neural network module, FeedForward:
class FeedForward(nn.Module):
    """
    The FeedForward module is a small neural netowkr consisting of two linear layers and a GELU activation function
    In the 124-million-parameter GPT model, it receives the input batches with tokens that have an embedding size of 769
    each via the GPT_CONFIG_124M dictionary
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(

            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

        )

    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(GPT_CONFIG_124M)
#   Creates sample input with batch dimension 2
x = torch.rand(2, 3, 768)
out = ffn(x)
#   result is torch.Size([2, 3, 768])
print(out.shape)

#   The FeedForward module plays a crucial role in enhancing the model's ability to learn from and generalize the data
#   It eternally expands the embedding dimension into a higher idmensional space through the first linear layer
#   Followed by a nonlinear GELU activation and then a contraction back to the original
#   dimension with the second linear transformation

#   Page 109
#   4.4 Adding shortcut connection
#   shortcut connections or skip / residual connections were origninall proposed for deep networks in computer vision to
#   mitigate the challenge of vanishing gradients
#   vanishing gradient problem refers to the issue where gradients (guide weight updates during training) become
#   progressively smaller as they propagate backward through the layers,
#   making it difficult to effectively main earlier layers
#   shortcut connection creates an alternative, shorter path for the gradient to flow through the network by skipping
#   one or more layers
#   achieved by adding the output of one layer to the ouput of a later layer
#   these connections are also

class ExampleDeepNeuralNetwork(nn.Module):
    """
    Implements a deep neural network with five layers, each consisting of a
    linear layer and a GELU activation function
    In the forward pass, iteratively pass the input through the layers and optionally add the
    shortcut connections if the self.use_shortcut attribue is set to True
    """
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        #   Implements five layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            #   Compute the output of the current layer
            layer_output = layer(x)
            #   Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

#   Use the code to initialize a neutral network without shortcut connections
#   Each layer will be initialized such that it accepts an example with three input values
#   and returns three output values
#   the last layer returns a single output value:

layer_sizes = [3, 3, 3, 3, 3, 1]
#   specifies random seed for the initial weights for reproducibility
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

#   implement a function that computes the gradients in the model's backward pass:
#   specifies a loss function that computes how close the model output and a
#   user-specified target are.
#   When calling loss.backward(), PyTorch computes the loss gradient for each layer in the model
#   Can iterate through the weight parameters via model.named_parameters
def print_gradients(model, x):
    #   Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    #   Calculates loss based on how close the target and output are
    loss = loss(output, target)

    #   Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


#   use the print_gradients function and apply it to the model without skip connections:
print_gradients(model_without_shortcut, sample_input)

#   the gradients become smaller as we progress from the last layer to the first layer
#   this is called vanishing gradient problem
#   instantiate a model with skip connections and see how it compares
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

#   Page 113
#   4.5 Connecting attention and linear layers in a transformer block
#   The transformer block is repeated a dozen times in the 124-million-parameter GPT-2 architecture, combines several concepts: multi-head
#   attetnion, layer normalization, dropout, feed forward layers, and GELU activations
#   When a transformer block processes an input sequence, each element in the sequqnce is represented by a fixed-size vector
#   the operations within the transformer block, including multi-head attention and feed forward layers are
#   designed to transform these vectors in a way that preserves their dimensionality
#   self-attention mechanism in the multi-head attention block identifies and analyzes relationships between elements in the input sequence.
#   The feed forward network modifies the data individually at each position.

#   Create the TransformerBlock:
class TransformerBlock(nn.Module):
    """
    Includes a multi-head attention mechnaism and a feed forward network
    Layer normalization is applied before each of these two components and
    dropout is applied after them to regularize the model and prevent overfitting
    This is known as Pre-LayerNorm
    Implements a forward pass, where each componenet is followed by a shortcut connection
    that adds the input of the block to its output
    This helps gradients flow thourgh the network during training and improves the learning
    of deep models
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        #   Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        #   Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        #   Adds the original input back
        x = x + shortcut
        return x

#   Instantiate a transformer block and feed it soe sample data:
#   The presentation of shape throughout the transformer block architecture is not incidental but a crucial aspect of its design
#   enables its effective application across a wide range of sequence-to-sequence tasks
#   each output vector directly corresponds to an input vector, maintaining a one-to-one relationship
#   Page 116
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

#   4.6 Coding the GPT model
#   Replace the DummyTransformerBlock and DummyLayerNorm placeholders with the real TransformerBlock and
#   LayerNorm classes
#   Assemble a fully working version of the original 124-million-parameter version of GPT-2

class GPTModel(nn.Module):
    """
    The TransformerBlock class allows the GPTModel class to be relatively small nad compact
    The __init__ constructor of this GPTModel class initializes the token and positinal embedding layers
    using the configurations passed in via a Python dictionary, cfg
    Embedding layers are responsible for converting input token indices into dense vectors and
    adding positional information

    __init__ method creates a sequential stack of TransformerBlock modules equal to the number of layers specified
    in cfg

    A LayerNorm layer is applied, standardizing the outputs from the transformer blocks ot stabilize the learning process

    A linear output head without bias is defined, which projects the transformer's output into the vocabulary space
    of the tokenizer to generate logits for each token in the vocabulary

    Forward method takes a batch of input token indices, computes their embeddings,
    appies the positional embeddings, passes the sequence through the transformer blocks,
    normalizes the final output, and then computes the logits, representing the next
    token's unnormalized probabilities
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range (cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        #   The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

#   Initialize the 124-million-parameter GPT model using the GPT_CONFIG_124M dictionary passed into the cfg parameter and fixed
#   it with the batch text input

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

#   Using the numel() method "number of elements" we can collect the total number of parameters in the model's parameter tensors:
total_params = sum(p.numel() for p in model.parameters())
#   The result is 163,009,536
print(f"Total number of parameters: {total_params:,}")

#   if we initalized 124 million parameter gpt why is the actual number of parameters 163 millions?
#   a concept called "weight tying" that was used in the original GPT-2 architecture. It reuses the weights
#   from the token embedding layer in its output layer
#   Take a look at the shapes of the token embedding layer and linear output layer that was initialized on the model via the GPTModel:
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# Remove the output layer parameter count from the total GPT-2 model count according to the weight tying:
#   Weight tying reduces the overall memory footprint and computational complexity of the model
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters()
                       )
)

#   The result is 124,412,160
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}")

#   Compute the memory requirements of the 163 million parameters in the GPTModel object:

#   Calculates the total size in bytes
total_size_bytes = total_params * 4
#   Converts to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

#   4.7 Generating Text
#   Implement the code that converts the tensor outputs of the GPT model back into text

#   idx is a (batch, n_token) array of indices in the current context
#   demonstrates a simple implementation of a generative loop for a language model using PyTorch
def generate_text_simple(model, idx,
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        #   Crops current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

            #   Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]
            #   probas has shape (batch, vocab_size)
            probas = torch.softmax(logits, dim=-1)
            #   idx_next has shape (batch, 1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            #   appends sampled index to the running sequence, where idx has shape (batch, N_tokens + 1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

#   try out the generate_text_simple function with the "Hello, I am" context as input
#   First, encode the input context into token IDs

start_context = "Hello, I Am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
#   adds batch dimension
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

#   Next, put the model into .eval() mode which disables random components like dropout that are only used during training
#   use the generate_text_simple function on the encoded input tensor:

#   disables the dropout since we are not training the model
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

#   output is Hello, I am Featureiman Byeswickattribute argue or some other gibberish
#   Haven't trained the model yet
decode_text = tokenizer.decode(out.squeeze(0).tolist())
print(decode_text)









#   C H A P T E R  F I V E  P A G E  1 2 8
#   P R E T R A I N I N G  O N  U N L A B E L E D  D A T A

#   learn basic model evaluation techniques to measure the quality of the generated text which is a requirement
#   for optimizing the LLM during the training process
#   how to load pretrained weights, giving the LLM a solid starting point for fine-tuning

#   Page 129
#   Evaluating generative text models
#   set up LLM for text generation
#   basic ways to evaluatee the quality of the generated text

#   Page 130
#   Using GPT to generate text
#   set up the LLM
#   initialize the GPT model

import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #   shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,    #   possible and common to set dropout to 0
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

#   after the training, will update the context size setting and load pretrained weights to work with a model configured
#   for a 1024-token context length setting and load pretrained weights to work with a model confiugred for a
#   1024-token context length

#   implement the text generation process:
import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    #   .unsqueeze(0) adds the batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    #   Removes batch dimension
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

#   returns Every effort moves you rentingetic wasn? refres RexMeCHicular stren
#   The model isn't predicting coherent text becuase it hasn't nudergone training
#   have to implement a numerical method to evaluate the generated content to monitor and enhance the model's performance
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#   5.1.2   Calculating the text generations loss
#   techniques for numerically assessing text quality generated during training
#   calculate a text generation loss

#   1.  the vocab to map the input text to token IDs 2. obtain seven-dimensional probability row vector for each input token via the softmax function
#   3.  Locate the index position with the highest probability value in each row vector which is done via the argmax function
#   4.  obtain all predicted token IDs, as the index position with the highest probabilities
#   5.  map index positions back into text via the vocab

#   Page 133
#   Consider these two input examples
#   Step 1
inputs = torch.tensor([[16833, 3626, 6100], #   "every effort moves"
                       [40, 1107, 588]])    #   "I really like"
#   matching these inputs, the targets contain the token IDs we want the model to produce:
targets = torch.tensor([[3626, 6100, 345],      #   "effort moves you"
                        [1107, 588, 11311]])   #   "really like chocolate"
#   targets are the inputs bhut shifted one position forward
#   shifting strategy is crucial for teaching the model to predict the next token in a sequence

#   feed the inputs into the model to calculate logits vectors for the two input examples, each computing three tokens:
#  apply softmax function to transform the logits into probability scores

#   Disables gradient training since we are not training yet
with torch.no_grad():
    logits = model(inputs)

#   Probability of each token in vocabulary
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

#   complete steps three and four by applying the argmax function to the probability scores to obtain the corresponding token IDs
token_ids = torch.argmax(probas, dim=-1)
print("Token IDs:", token_ids)

#   Step five: convert the token IDs back into text:

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

#   now want to evaluate the performance of the model's generated text numerically via a loss
#   useful for measuring the quality of the generated text
#   alos a building block for implementing the training function

#   for each of the two input texts, print the initial softmax probability scores corresponding to the target tokens'

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

#   goal of training an LLM is to maximize the likelihood of the correct token
#   increase probability relative to other tokens
#   ensure the LLM consistently picks the target token--essentially the next word in the sentence -- as the next token it generates

#   next, calculate the loss for the probability scores of the two example batches
#   target_probas_1 and target_probas_2
#   proceed with step four, applying the logarithm to the probability scores:
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

#   combine the probabilities into a single score by computing the average (step five of calculating the loss)
avg_log_probas = torch.mean(log_probas)
#   resulting average log probability score is tensor(-10.7940)
print(avg_log_probas)

#   goal is to get the average log probability as close to 0 as possible by updating the model's weight
#   as part of the training process
#   in deep learning the common practice is to bring the negative average log probability down to 0
#   the negative average log probability is simply the average log probability multiplied by -1
#   corresponds to step 0

#   in deep learning the process of turning the negative value into the positive value is known as cross entropy
#   PyTorch has a cross entropy function
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

#  right now, the logits tensor has three dimensions batch size, number of tokens, and vocabulary size. but the targets tensor has two dimensions batch size and number of tokens
#   For the cross entropy loss function in PyTorch, have to flatten these tensors of tokens

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
#   logits contain the unscaled model outputs before they enter the softmax function to obtain the probability scores
print("Flattened logits:", logits_flat.shape)
#   targets are the token IDs that the LLM generate
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

#   we have now calculated the loss for two small text inputs for illustration purposes
#   apply the loss computation to the entire training and validation sets

#   to compute the loss on the training and validation datases, use a very small text dataset, the "The Verdict."
#   by selecting a text from the public domain, we circumvent any related to usage rights
#   using such a small dataset allows for the execution of code examples on a standard laptop computer in a matter of
#   minutes, even without a high-end GPU

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

#   check the number of characters and tokens in the dataset:

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
#   output is 20479
print("Characters:", total_characters)
#   output is 9145
print("Tokens:", total_tokens)

#   with just 5,145 tokens, the text might seem too small to train an LLM (this is for training purposes)
#   later will load pretrained weights from OpenAI into our GPTModel code

#   next, divide the dataset into a training and a validation set and use the data loaders from chapter 2 to prepare
#   the batches for LLM training
#   due to spacial constraints, we use a max_length=6
#   for the actual data loaders, set the max_length equal  to the 256-token context length that the LLM supports
#   so that the LLM sees longer texts during training
#   are training the moel with training data presented in similarly sized chunks for simplicity and efiiency

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

#   using the train_data and val_data subsets, can now create the respective data loader reusing the create_dataloader_v1

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

#   as an optional check, we can iterate through the data loaders to ensure that they were created correctly

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

#   next, implement a utility function to calculate the cross entropy loss of a given batch, to implement
#   the following calc_loss_loader function that computes the loss over all the batches sampled by a
#   given data loader
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

#   we can now use this calc_loss_batch utility function
#   computes the loss for a single batch, to implement the following calc_loss_loader function that computes the loss over
#   all the batches sampled by a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
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

#   by default, the calc_loss_loader function iterates over all batches in a given data loader
#   accumulates the loss in the total_loss variable
#   computes and averages the loss over the total number of batches.
#   alternatively, can specify a smaller number of batches via num_batches to speed up the evaluation during model training

#   if you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#   Disables gradient tracking for efficiency because we are not training, yet
with torch.no_grad():
    #   Via the "device" setting, we ensure the data is loaded onto the same device as the LLM model
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

#   the loss values are relatively high because the odel has not yet been trained
#   the loss approaches 0 if the model learns to generate the next tokens as they appear in the training and validation sets

#   now train the LLM to reduce this loss so that it becomes better at generating text
#   focus on pretraining the LLM
#   After model training, implement alternative text generation strategies and save and load pretrained
#   model weights

#   Page 146
#   5.2 Training an LLM
#   implement the code for pretraining the LLM, GPTModel
#   straight forward training loop
#   typical PyTorch neural network training workflow
#   step 1: Iterate over training epochs
#   step 2: Iterate over batches in each training epoch
#   step 3: Reset loss gradients from previous batch iteration
#   step 4: Calculate loss on current batch
#   step 5: backward pass to calculate loss gradient
#   step 6: Update model weights using loss gradients
#   step 7: print training and validation set losses
#   step 8: Generate sample text for visual inspection

#   implement this training flow via the train_model_simple function

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    #   Initializes lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    #   Starts the main training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            #   Resets loss gradient from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            #   Calculates loss gradients
            loss.backward()
            #   updates model weights using loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            #   Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}"
                )

        #   Prints a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    #   Dropout is disabled during evaluation for stable, reproducible results
    model.eval()
    #   Disables gradient tracking, which is not required during evaluation, to reboot the computational overhead
    with torch.no_grad():
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    #   Compact print format
    print(decoded_text.replace("\n", " "))
    model.train()

#   train a GPTModel instance for 10 epochs using an AdamW optimizer and the train_model_simple function

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    #   The .parameters() method returns all trainable weight parameters of the model
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

#   create a simple plot that shows the training and validation set losses side by side
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    #   Creates a second x-axis that shares the same y-axis
    ax2 = ax1.twiny()
    #   Invisible plot for aligning ticks
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

#   memorization is expected since we are working with a very, very small training dataset and training the model for multiple epochs.
#   it is common to train a model on a much larger dataset for only one epoch

#   have completed four of our objectives:
#   text generation, text evaluation, training & validation losses, and LLM training function

#   next is text generation strategies for LLMs to reduce training data memorization and increase the originality of the LLM-generated text
#   before we cover weight loading and saving and loading pretrained weights from OpenAI's GPT model
#   Page 51
#   5.3 Decoding strategies to control randomness

#   text generation strategies or decoding strategies are used to generate more original through temperature scaling and top-k sampling
#   first, transfer the model from the GPU to CPU since interference with a relatively small model does not require a GPU
#   after training we put the model into evaluation mode to turn off random components such as dropout

model.to("cpu")
model.eval()
#   plug the GPTModel instance (model) into the generate_text_simple function, which uses the LLM to generate one token at a time

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#   the generated token is selected at each generation step corresponding to the largest probability score among all tokens in the vocabulary.
#   the LLM will always generate the same outputs even if we run the preceding generate_text_simple function multiple times on the same start context
#   (Every effort moves you).

#   5.3.1   Temperature scaling
#   temperature scaling adds a probabilistic selection process to the next-token generation task.
#   to generate txt with more variety, we can replace argmax with a function that samples from a probability distribution

#   briefly discuss the next-token generation process using a very small vocabulary for illustration purposes

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

#   next assume the LLM is given the start context "every effort moves you" and generates the following
#   next-token logits
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

#   inside generate_text_simple convert the logits into probabilities via the softmax function and obtain
#   the token ID corresponding to the generated token via the argmax function
#   then map them bak into text via the inverse vocabulary

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

#   since the largest logit value and, correspondingly, the largest softmax probabiity score are in the fourht position,
#   the generated word is "forward"
#   to iiplement a probabilistic sampling process, replace argmax with the multinomial function in PyTorch
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
#   the printed output is "forward" just like before. What happened?
#   the multinomial function samples the next token prportional to its probabilit score: "forward" is still the most likely token and
#   will be selected by multinomial most of the time but not all the time
print(inverse_vocab[next_token_id])

#   to illustrate this, implement a function that repeats this smapling 1,000 times:
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_tokens(probas)

#   if we replaced the argmax function with the multinomial function inside the generate_and_print_sample function, the LLM would sometimes generate texts such
#   as every effort moves you toward, every effort moves you inches, and every effort moves you closer instead of
#   every effort moves you forward

#   can further control the distribution and selection process via a concept called temperature scaling
#   description for dividing the logits by a number greater than 0
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

#   temperatures greater than 1 result in more uniformly distributed token probabilities and temperatures smaller than 1 will
#   result in more confident distributions
#   illustrate by plotting the original probabilities alongside probabilities scaled with different temperature values
#   Original, lower and higher confidence
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + 1 * bar_width, scaled_probas[i],
                   bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

#   a temperature of 1 divides the logits by 1 before passing them to the softmax function to compute the probability scores
#   using a temperature of 1 is the same as not using any temperature scaling
#   applying very small temperatures, such as 0.1, will result in sharper distributions such that the behavior
#   of the multinomial function selects the most likely token almost 100% of the time

#   5.3.2   Top-k sampling
#   higher temperature values result in more uniformly distributed next-token probabilities, which result
#   in more diverse outputs as it reduces the likelihood of the model repeatedly selecting the most probable token
#   method allows for the exploring of less likely but potentially more interesting and creative paths in the generation process.
#   sometimes leads to grammatically incorrect or completely nonsensical outputs such as every effort moves you pizza
#   Top-k sampling when combined with probabilistic sampling and temperature scaling can imprvoe the text geenration results.
#   can restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from the selection process by masking their probability scores
#   replaces all nonselected ogits with negative infinity value (-inf), such that when computing the softmax values,
#   the probability scores of the non-top-k  tokens are 0
#   and the remaining probabilities sum up to 1
#   in code we can implement the top-k procedure as follows:
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

#   then, apply PyTorch's where function to set the logit values of tokens that are below the lowest logit value within our
#   top-three selection to negative infinity (-inf)

new_logits = torch.where(
    #   identifies logits less than the minimum in the top 3
    condition=next_token_logits < top_logits[-1],
    #   Assigns -inf to these lower logits
    input=torch.tensor(float('-inf')),
    #   Retains the original logits for all other tokens
    other=next_token_logits
)
print(new_logits)

#   lastly, apply the softmax function to turn these into next-token probabilities
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

#   can now apply the temperature scaling and multinomial function for probabilistic sampling to select the next token among these
#   three non-zero probability scores to generate the text token
#   do this by modifying the text-generation function

#   5.3.3   Modifying the text generation function
#   combine temperature sampling and top-k sampling to modify the generate_text_simple function
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    #   The for loop is the same as before: gets logits and only focuses on the last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        #   Filters logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        #   Applies temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        #   Carries out greedy next-token selection as before when temperature scaling is disabled
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        #   Stops generating early if end-of-sequence token is encountered
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#   5.4 Loading and saving model weights in PyTorch
#   pretraining LLMs is computationally expensive
#   It is important to be able to save the LLM so that we don't have to rerun the training every time we want to use it in a new session
#   how to save and load a pretrained model
#   recommended way is to save a model's state_dict, a dictionary mapping each layer to its parameters, using the torch.save function
torch.save(model.state_dict(), "model.pth")
#   "model.pth" is the filename where the stats_dict is saved
#   the .pth extension is a convention for PyTorch files, though could use any file extension
#   after saving the model weights via the state_dict, can load the model weights into a new GPTModel model instance

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
#   using model.eval() switches the model to evaluation mode for inference, disabling the dropout layers of the model
#   want to continue pretraining a model state later, saving the optimizer state is also recommended
model.eval()

#   AdamW uses historical data to adjust learning rates for each model parameter dynamically.
#   without it, the optimizer resets, and the model may learn suboptimally or even fail to conver properly
#   it will lose the ability to generate coherent text
#   using torch.save, we can save both the model and optimizer state_dict contents:
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)
#   then, we can restore the model and optimizer states by first loading the saved data via torch.load
#   and then using the load_state_dict method

checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();


#   5.5 Loading pretrained weights from OpenAI

#   OpenAI opnely shared the weights of their GPT-2 models, eliminating the need to invest tens to hundreds of thousands of dollars in
#   retraining the model on a large corpus ourselves
#   load these weights into the GPTModel class and use the model for text generation
#   weights refers to the weight parameters stored in the .weight attributes of PyTorch's Linear and Embedding layers
#   OpenAI originally saved the GPT-2 weights via TensorFlow which we have to install to load the weights in Python
#   code will use a progress bar tool called tqdm to track the download process
#   pip install tensorflow

import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

#   next after downloading this file to the local directory of the Python session,
#   briefly inspect the contents of this file to ensure that it was saved correctly and
#   contains valid Python code
#   can now import the download_and_load_gpt2 function from the gpt_download.py file which will load the
#   GPT-2 architecture settings and weight parameters into our Python session:

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

#   inspect the contents of settings and params:

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

#   printing the weight contents would take up too much screen space
#   can inspect these weight tensors by printing the whole dictionary via print(params) or by
#   selecting individual tensors via the respective dictionary keys:

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

#   we downloaded and loaded the weights of the smallest GPT-2 model via the download_annd_load_gpt2(model_size="124M", ...) setting.
#   OpenAI also shares the weights of larger models: 355M, 774M, and 1558M.
#   The overall architecture is the same

#   need to transfer the weights from the settings and params dictionaries into the GPTModel instance
#   first create a dictionary that lists the differences between the different GPT model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}