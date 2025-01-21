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

# ship model to device and set to eval mode
model.to(device)
model.eval()


def get_next_most_likely_tokens(logits, top_k=20):
    probs = F.softmax(logits, dim=-1)

    return torch.topk(probs, top_k)


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

    # clean run
    y, last_token_logits = model.generate(clean, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_logits=True, patch_config=None)

    clean_expected_index = tokenizer(' ' + clean_expected_next).to(device)[0]
    clean_expected_prob = last_token_logits[0][clean_expected_index].item()

    # initialize a difference matrix to hold the differences of the last logits between clean and corrupted runs
    logits_diff = torch.zeros((12, clean.size()[1]))

    # corrupted run
    for i in range(12):
        # iterate over all tokens in the input sequence
        for j in range(clean.size()[1]):
            _, last_token_logits_corr = model.generate(corr, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_logits=False, patch_config=(i, j))
            corr_expected_index = tokenizer(' ' + corr_expected_next).to(device)[0]
            corr_expected_prob = last_token_logits_corr[0][corr_expected_index].item()
            logits_diff[i][j] = corr_expected_prob - clean_expected_prob


    probs, indices = get_next_most_likely_tokens(last_token_logits, top_k=20)

    # decode the tokens and print them with their probabilities
    print("CLEAN - MOST LIKELY TOKENS")
    for id, (prob, idx) in enumerate(zip(probs[0], indices[0])):
        idx_tensor = torch.tensor([idx.item()])
        token = tokenizer.decode(idx_tensor)
        print(f"{id + 1}. Token Index: {idx.item()}, Token: {token}, Probability: {prob.item()}")

    _, last_token_logits_corr = model.generate(corr, max_new_tokens=steps, do_sample=do_sample, top_k=15, save_logits=False)
    probs_corr, indices_corr = get_next_most_likely_tokens(last_token_logits_corr, top_k=20)

    # decode the tokens and print them with their probabilities
    print("CORRUPTED - MOST LIKELY TOKENS")
    for id, (prob, idx) in enumerate(zip(probs_corr[0], indices_corr[0])):
        idx_tensor = torch.tensor([idx.item()])
        token = tokenizer.decode(idx_tensor)
        print(f"{id + 1}. Token Index: {idx.item()}, Token: {token}, Probability: {prob.item()}")

    # decode input tokens
    input_tokens = [tokenizer.decode(torch.tensor([token])) for token in clean.squeeze().tolist()]

    # Visualize results with token labels
    plt.figure(figsize=(30, 16))
    sns.heatmap(logits_diff, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=input_tokens, yticklabels=[f'Layer {i}' for i in range(12)])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.title("Activation Patching Heatmap")
    plt.tight_layout()
    plt.savefig(f'{output_name}.png')
    plt.close()
