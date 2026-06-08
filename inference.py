import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder
import argparse

# --- Model Hyperparameters (Must exactly match the training notebook) ---
VOCAB_SIZE = 4096
CONTEXT_LENGTH = 256
N_EMBD = 256
N_HEAD = 8
N_LAYER = 4
DROPOUT = 0.0 # Set to 0.0 for inference

MODEL_PATH = "tinystories_model.pt"
VOCAB_PATH = "tinystories_bpe.json"

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.register_buffer("bias", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH))
                                     .view(1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyStoriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        """
        Autoregressively generates new tokens one by one.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the maximum sequence length (CONTEXT_LENGTH)
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            
            # Forward pass to get predictions
            logits, _ = self(idx_cond)
            
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Break out of the loop if the model generates <eos>
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                idx = torch.cat((idx, idx_next), dim=1)
                break
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ==========================================
# INFERENCE SCRIPT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate stories using trained TinyStories model.")
    parser.add_argument('--prompt', type=str, default="", help="Starting prompt for the model")
    parser.add_argument('--tokens', type=int, default=200, help="Number of new tokens to generate")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature for sampling (higher = more random)")
    parser.add_argument('--top_k', type=int, default=10, help="Top-K sampling limit")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.from_file(VOCAB_PATH)
        # Add post-processor to prepend <bos> like we did in training, 
        # but NOT <eos> because we want it to keep generating from the prompt.
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A",
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>"))
            ],
        )
    except Exception as e:
        print(f"Failed to load '{VOCAB_PATH}'. Please ensure it's in the same directory.")
        return

    # Load Model
    print("Loading model...")
    model = TinyStoriesModel().to(device)
    try:
        # load weights_only=True for safety
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode (turns off dropout)
    except Exception as e:
        print(f"Failed to load '{MODEL_PATH}'. Please ensure it's in the same directory. Error: {e}")
        return

    # Encode Prompt
    print(f"\nPrompt: \"{args.prompt}\"")
    print("Generating...")
    
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    eos_id = tokenizer.token_to_id("<eos>")

    # Generate Output
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=args.tokens, 
        temperature=args.temp, 
        top_k=args.top_k,
        eos_token_id=eos_id
    )

    # Decode and print
    # We grab the first item in the batch ([0]) and convert to a regular python list
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    
    # Clean up the unwanted spaces around punctuation marks
    generated_text = generated_text.replace(" ,", ",")
    generated_text = generated_text.replace(" .", ".")
    generated_text = generated_text.replace(" !", "!")
    generated_text = generated_text.replace(" ?", "?")
    generated_text = generated_text.replace(" ' ", "'")
    generated_text = generated_text.replace(" :", ":")
    generated_text = generated_text.replace(" ;", ";")
    generated_text = generated_text.replace(" \" ", " \"")

    print(generated_text)

if __name__ == '__main__':
    main()

