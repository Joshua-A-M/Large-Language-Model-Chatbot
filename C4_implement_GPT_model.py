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

from C3_coding_attention_mechanisms import MultiHeadAttention
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