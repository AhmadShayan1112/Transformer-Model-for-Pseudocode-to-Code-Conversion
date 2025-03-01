import json
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import json
with open('vocabulary.json', 'r', encoding='utf-8') as f:
    vocab_dict = json.load(f)
Vocab_size = len(vocab_dict)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_len=60):
        super(TransformerModel, self).__init__()

        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )

        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def generate_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(device)

    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = self.generate_mask(tgt.size(1))

        src_key_padding_mask = src == 0
        tgt_key_padding_mask = tgt == 0

        src_emb = self.positional_encoding(self.encoder_embedding(src))
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))

        output = self.transformer(
            src_emb.permute(1, 0, 2), 
            tgt_emb.permute(1, 0, 2),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output.permute(1, 0, 2))
        return output
input_vocab_size = Vocab_size  
output_vocab_size = Vocab_size  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_vocab_size, output_vocab_size).to(device)

with open("vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
token2idx = vocab
idx2token = {idx: token for token, idx in vocab.items()}
model_2 = TransformerModel(input_vocab_size, output_vocab_size).to(device)  
optimizer = torch.optim.Adam(model_2.parameters(), lr=0.001)
def load_model(model_path, model_2, optimizer):
    checkpoint = torch.load(model_path, map_location=device)
    model_2.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model_2.eval()  
    print(f"✅ Model loaded from {model_path}")
load_model("transformer_model.pt", model_2, optimizer)
def tokenize(sentence, token2idx):
    """Tokenizes the input sentence using the vocabulary."""
    return [token2idx.get(token, token2idx["<UNK>"]) for token in sentence.split()]

def detokenize(indices, idx2token):
    """Converts token indices back to words."""
    return " ".join([idx2token.get(idx, "<UNK>") for idx in indices])
def predict(model, sentence, max_len=60):
    """Generates C++ code from pseudocode or vice versa."""
    model.eval()
    input_tokens = tokenize(sentence, token2idx)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)
    output_tokens = [token2idx["<start>"]]
    
    with torch.no_grad():
        for _ in tqdm(range(max_len), desc="Decoding", leave=True):
            output_tensor = torch.tensor(output_tokens).unsqueeze(0).to(device)
            predictions = model(input_tensor, output_tensor)  
            next_token = predictions[0, -1].argmax(dim=-1).item()
            if next_token == token2idx["<end>"]:
                break
            output_tokens.append(next_token)
    return detokenize(output_tokens[1:], idx2token)
# manual_input = 'function to add two numbers'
# output = predict(model_2, manual_input)
# print("\nGenerated Output:\n", output)

def clean_spaces(text):
    lines = text.split("\n")  # Split into lines
    cleaned_lines = [" ".join(line.split()) for line in lines]  # Clean spaces in each line
    return "\n".join(cleaned_lines)  # Join lines back with newlines
def format_lines(parameter1=model_2, parameter2=None):
    lines = parameter2.split("\n")  # Split the provided text into lines
    formatted_lines = [f"{predict(model_2, line)}" for i, line in enumerate(lines)]  
    return "\n".join(formatted_lines)  # Join formatted lines with newlines  # Join formatted lines with newlines
#output_text = format_lines(input_text)