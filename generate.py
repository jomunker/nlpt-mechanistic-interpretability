import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

set_seed(3407)

use_mingpt = True
model_type = 'gpt2'
device = 'cpu'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id  # suppress a warning

# ship model to the device and set to eval mode
model.to(device)
model.eval()
layer_count = len(model.transformer.h)


def get_next_most_likely_tokens(logits, top_k=20):
    probs = F.softmax(logits, dim=-1)

    return torch.topk(probs, top_k)


def get_token_probability(expected_next_token, last_token_logits, tokenizer):
    idx = tokenizer(' ' + expected_next_token).to(device)[0]
    prob = last_token_logits[0][idx].item()
    return prob


def print_next_token_probs(last_token_logits, tokenizer, top_k=20):
    probs, indices = get_next_most_likely_tokens(last_token_logits, top_k=top_k)
    for id, (prob, idx) in enumerate(zip(probs[0], indices[0])):
        idx_tensor = torch.tensor([idx.item()])
        token = tokenizer.decode(idx_tensor)
        print(f"{id + 1}. Token Index: {idx.item()}, Token: {token}, Probability: {prob.item()}")


def generate(prompt_clean='', clean_expected_next='', prompt_corr='', corr_expected_next='', num_samples=1, steps=1, do_sample=True, output_name="heatmap"):
    # tokenize the input prompt into integer input sequence

    tokenizer = BPETokenizer()
    if prompt_clean == '':
        # to create unconditional samples...
        # manually create a tensor with only the special <|endoftext|> token
        # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
        clean = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long).to(device)
        corr = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long).to(device)
    else:
        clean = tokenizer(prompt_clean).to(device)
        corr = tokenizer(prompt_corr).to(device)

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    clean = clean.expand(num_samples, -1)

    # CLEAN RUN
    y, last_token_logits = model.generate(clean, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_activations=True, patch_config=None)
    clean_expected_token_prob = get_token_probability(clean_expected_next, last_token_logits, tokenizer)

    # decode the next most likely tokens for the clean run and print them with their probabilities
    print("CLEAN - MOST LIKELY TOKENS")
    print_next_token_probs(last_token_logits, tokenizer)

    # initialize a difference matrix to hold the differences of the next expected token probabilities
    probs_diff = torch.zeros((layer_count, clean.size()[1]))

    # corrupted run with activation patching
    for i in range(layer_count):
        # iterate over all tokens in the input sequence
        for j in range(clean.size()[1]):
            _, last_token_logits_corr = model.generate(corr, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_activations=False, patch_config=(i, j))
            corr_expected_token_prob = get_token_probability(corr_expected_next, last_token_logits_corr, tokenizer)
            probs_diff[i][j] = corr_expected_token_prob - clean_expected_token_prob

    # decode input tokens to strings
    input_tokens = [tokenizer.decode(torch.tensor([token])) for token in clean.squeeze().tolist()]

    # Visualize results with token labels
    plt.figure(figsize=(30, 16))
    sns.heatmap(probs_diff, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=input_tokens, yticklabels=[f'Layer {i}' for i in range(12)])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.title("Activation Patching Heatmap")
    plt.tight_layout()
    plt.savefig(f'./heatmaps/{output_name}.png')
    plt.close()

    # CORRUPTED RUN
    _, last_token_logits_corr = model.generate(corr, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_activations=False)

    # decode the next most likely tokens for the corrupted run and print them with their probabilities
    print("CORRUPTED - MOST LIKELY TOKENS")
    print_next_token_probs(last_token_logits_corr, tokenizer)
