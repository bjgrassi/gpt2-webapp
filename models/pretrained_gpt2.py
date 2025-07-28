'''
[259641900] Gabriel Amorim Moterani
[249578450] Bruna Jacinto Grassi

COSC5437002 - Neural Networks and Deep Learning
Prof. Syed Muhammad Danish
Algoma University

Youtube Link:
'''
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import psutil
import time
from typing import List, Dict

@dataclass
class MeasurementResult:
    component_name: str
    memory_usage_mb: float
    cpu_usage_percent: float
    inference_time_ms: float

def measure_component(model: torch.nn.Module, input_tensor: torch.Tensor, component_name: str) -> MeasurementResult:
    """
    Measure memory, CPU usage, and inference time of a specific component.
    """
    process = psutil.Process()

    # Warm-up (ensure stable measurements)
    with torch.no_grad():
        _ = model(input_tensor)

    # Memory measurement
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    cpu_before = process.cpu_percent(interval=None)

    start_time = time.perf_counter()

    with torch.no_grad():
        _ = model(input_tensor)

    inference_time_ms = (time.perf_counter() - start_time) * 1000
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB

    return MeasurementResult(
        component_name=component_name,
        memory_usage_mb=mem_after - mem_before,
        cpu_usage_percent=cpu_after - cpu_before,
        inference_time_ms=inference_time_ms
    )

def measure_gpt2_components(model: torch.nn.Module, input_tensor: torch.Tensor) -> List[MeasurementResult]:
    """
    Measure each transformer block and feed-forward components in GPT-2.
    """
    results = []

    # Measure embedding layer
    embedding_layer = model.transformer.wte
    results.append(measure_component(embedding_layer, input_tensor, "Embedding Layer"))

    # Measure positional embeddings
    pos_embeddings = model.transformer.wpe
    pos_tensor = torch.arange(input_tensor.size(1), device=input_tensor.device)
    results.append(measure_component(pos_embeddings, pos_tensor, "Positional Embeddings"))

    # Measure each transformer block
    for i, block in enumerate(model.transformer.h):
        # Clone input to avoid modifying original
        x = input_tensor.clone()

        # Measure attention component
        attn_result = measure_component(block.attn, block.ln_1(x), f"Block {i} - Attention")
        results.append(attn_result)

        # Measure MLP component
        mlp_result = measure_component(block.mlp, block.ln_2(x), f"Block {i} - MLP")
        results.append(mlp_result)

    # Measure final layer norm and head
    final_ln = model.transformer.ln_f
    lm_head = model.lm_head
    x = input_tensor.clone()
    results.append(measure_component(final_ln, x, "Final LayerNorm"))
    results.append(measure_component(lm_head, final_ln(x), "LM Head"))

    return results

def print_measurements(results: List[MeasurementResult]):
    """Print measurement results in a formatted table."""
    print("\nComponent-wise Measurements:")
    print("-" * 90)
    print(f"{'Component':<30} | {'Memory (MB)':>12} | {'CPU Usage (%)':>12} | {'Time (ms)':>12}")
    print("-" * 90)

    for result in results:
        print(f"{result.component_name:<30} | {result.memory_usage_mb:>12.2f} | "
              f"{result.cpu_usage_percent:>12.2f} | {result.inference_time_ms:>12.4f}")
    print("-" * 90)

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        # https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"PRE-TRAINED: using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = DataLoaderLite(B=4, T=32)

    # get logits
    model = GPT(GPTConfig())
    model.to(device)

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_steps = 100 # Adjust as needed
    for i in range(num_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")

# Text Generation Function
def generate_text(
    model, 
    prompt, 
    max_length=50,  # Increased for longer completions
    temperature=0.7,  # Better default for creative tasks
    top_k=50,
    stop_tokens=None,  # Additional tokens to stop generation
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate text completions for any given prompt.
    
    Args:
        model: The GPT model
        prompt: Input text prompt (str)
        max_length: Maximum tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: Top-k sampling
        stop_tokens: List of token ids that stop generation (e.g., [13] for newline)
        device: Device to run on
    """
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt and convert to tensor
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate tokens
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            # Truncate if sequence exceeds block_size
            idx_cond = generated if generated.size(1) <= model.config.block_size else generated[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop conditions
            if stop_tokens and next_token.item() in stop_tokens:
                break
    
    # Decode and clean output
    full_text = enc.decode(generated[0].tolist())
    
    # Remove prompt if you only want the completion
    completion = full_text[len(prompt):].strip()

    # Create proper input for measurements
    # Use long integers for embedding indices
    input_tensor = torch.randint(0, model.config.vocab_size, (1, 32), dtype=torch.long, device=device)
    
    # Get position indices
    pos_tensor = torch.arange(0, 32, dtype=torch.long, device=device)
    
    # Measure components
    measurements = []
    
    # Measure embedding layer
    with torch.no_grad():
        tok_emb = model.transformer.wte(input_tensor)  # (1, 32, n_embd)
        pos_emb = model.transformer.wpe(pos_tensor)  # (32, n_embd)
        x = tok_emb + pos_emb.unsqueeze(0)  # (1, 32, n_embd)
    
    # Measure embedding components
    measurements.append(measure_component(model.transformer.wte, input_tensor, "Embedding Layer"))
    measurements.append(measure_component(model.transformer.wpe, pos_tensor, "Positional Embeddings"))
    
    # Measure each transformer block
    for i, block in enumerate(model.transformer.h):
        # Clone input to avoid modifying original
        x_clone = x.clone()
        
        # Measure attention component
        ln1_out = block.ln_1(x_clone)
        measurements.append(measure_component(block.attn, ln1_out, f"Block {i} - Attention"))
        
        # Measure MLP component
        ln2_out = block.ln_2(x_clone)
        measurements.append(measure_component(block.mlp, ln2_out, f"Block {i} - MLP"))
        
        # Update x for next block
        x = block(x_clone)
    
    # Measure final components
    measurements.append(measure_component(model.transformer.ln_f, x, "Final LayerNorm"))
    measurements.append(measure_component(model.lm_head, model.transformer.ln_f(x), "LM Head"))
    
    print_measurements(measurements)
    print(f"Prompt: '{prompt}'\nCompletion: '{completion}'\n")
    return completion, measurements