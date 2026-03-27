#   C H A P T E R  3
#   C O D I N G  A T T E N T I O N  M E C H A N I S M S
import torch
import torch.nn as torch
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
