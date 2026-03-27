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

from C2_work_with_text_data import create_dataloader_v1


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #   shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,    #   possible and common to set dropout to 0
    "qkv_bias": False
}

from C4_implement_GPT_model import GPTModel

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

from C4_implement_GPT_model import generate_text_simple

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

#   this is how to load the smallest model
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

#   update the NEW_CONFIG with the GPT2 1,024-token length
NEW_CONFIG.update({"context_length": 1024})

#   OpenAI used bias vectors in the multi-head attention module's linear layers to implement the query, key
#   and value matrix computations
NEW_CONFIG.update({"qkv_bias": True})


#   use the update NEW_CONFIG dictionary to initialize a new GPTModel instance
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

#   by default the GPTModel is initialized with random weights for pretraining
#   oevrride the random weights with the weights that were loaded into the params dictionary
#   first, define a small assign utility function that checks whether two tensors or arrays
#   have the same dimensions or shape and returns the right tensor as trainable PyTorch parameters:
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         f"Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))

#   next, define a load_weights_into_gpt function that loads the weights from the
#   params dictionary into a GPTModel instance gpt

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    #   Iterates over each transformer block in the model
    for b in range(len(params["blocks"])):
        #   The np.split function is used to divide the attention and bias weights into three
        #   equal parts for the query, key, and value components
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])['w'], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


#   In the load_weights-into_gpt function, we carefully match the weights from OpenAI's
#   implementation with our GPTModel implementation
#   OpenAI stored the weight tensor for the output projections layer for the first
#   transformer block as params["blocks"][0]["attn"]["c_proj"]["w"]
#   this weight tensor corresponds to gpt.trf_blocks[b].att.out_proj.weight, where gpt is a GPTMpodel instance
#   If we made a mistake in this function, we would notice this as the resulting GPT model would be unable
#   to produce coherent text

#   try the load_weights_into_gpt out in practice and load the OpenAI model weights into our
#   GPTModel instance gpt:
load_weights_into_gpt(gpt, params)
gpt.to(device)

#   if the model is loaded correctly, can now use it to generate new text
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))