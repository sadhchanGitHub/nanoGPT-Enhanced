# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    """A simple Bigram Language Model.

    Each token directly predicts the logits for the next token using a lookup table.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table maps each token to logits over vocab
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor of shape (B, T) with token indices.
            targets (torch.Tensor): Optional. Same shape (B, T), ground-truth next tokens.

        Returns:
            logits (torch.Tensor): Predictions of shape (B, T, vocab_size).
            loss (torch.Tensor or None): Cross-entropy loss if targets provided, else None.
        """
        # Get the initial logits, this is the model's true output
        logits = self.token_embedding_table(idx)  # Shape: (B, T, vocab_size)
    
        if targets is None:
            loss = None
        else:
            # Reshape for loss calculation, but do not overwrite the original logits variable
            B, T, C = logits.shape
            logits_for_loss = logits.view(B * T, C)
            #The modification of logits for the loss calculation should be a local operation. 
            #The method should always return the logits in their original, interpretable shape.
            targets_for_loss = targets.view(B * T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
    
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate new tokens.

        Args:
            idx (torch.Tensor): Tensor of shape (B, T), starting context.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: (B, T + max_new_tokens) containing the full sequence.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)         # (B, T, vocab_size)
            logits = logits[:, -1, :]     # last step → (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # turn into probabilities
            next_idx = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat([idx, next_idx], dim=1)  # append
        return idx

class MLPBigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(vocab_size, hidden_dim)  # input = one-hot
        self.fc2 = nn.Linear(hidden_dim, vocab_size)  # output = logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # One-hot encode inputs: (B, T, vocab_size)
        x_onehot = F.one_hot(idx, num_classes=self.vocab_size).float()

        # Pass through MLP
        x = self.fc1(x_onehot)       # (B, T, hidden_dim)
        x = F.relu(x)
        logits = self.fc2(x)         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross entropy
            logits_flat = logits.view(B * T, self.vocab_size)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)           # (B, T, vocab_size)
            logits = logits[:, -1, :]       # last token
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx




# ngram=2 → bigram, ngram=3 → trigram, etc.
import torch
import torch.nn as nn
import torch.nn.functional as F

# ngram=2 → bigram, ngram=3 → trigram, etc.
class MLPNgramLanguageModel(nn.Module):
    """
    General N-gram MLP Language Model.
    ngram = 2 → bigram
    ngram = 3 → trigram
    etc.
    """
    def __init__(self, vocab_size, ngram=3, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram = ngram
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * ngram, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T < self.ngram:
            pad_len = self.ngram - T
            pad = idx[:, :1].repeat(1, pad_len)
            idx = torch.cat([pad, idx], dim=1)
            T = idx.size(1)

        # create contexts
        contexts = torch.stack([idx[:, i:i+self.ngram] for i in range(T - self.ngram + 1)], dim=1)
        x = self.token_embedding(contexts)
        B, num_ctx, n, E = x.shape
        x = x.view(B, num_ctx, n * E)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                         # (B, num_ctx, vocab_size)

        loss = None
        if targets is not None:
            if targets.size(1) < self.ngram:
                pad_len = self.ngram - targets.size(1)
                pad = targets[:, :1].repeat(1, pad_len)
                targets = torch.cat([pad, targets], dim=1)
            # align targets with contexts
            targets_shifted = targets[:, self.ngram - 1 : self.ngram - 1 + num_ctx]
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets_shifted.reshape(-1)
            )

        return logits, loss



    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            # --- Handle short starting sequences ---
            if idx.size(1) < self.ngram:
                pad_len = self.ngram - idx.size(1)
                pad = idx[:, :1].repeat(1, pad_len)
                context = torch.cat([pad, idx], dim=1)
            else:
                context = idx[:, -self.ngram:]

            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            # --- Top-k filtering ---
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(1)
                logits[logits < min_v] = -float('inf')

            # --- Sample next token ---
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx

class MLPNgramLanguageModel_posemd(nn.Module):
    def __init__(self, vocab_size, ngram=3, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram = ngram
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(ngram, embed_dim)  # positional embeddings
        
        self.fc1 = nn.Linear(embed_dim * ngram, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T < self.ngram:
            pad_len = self.ngram - T
            pad = idx[:, :1].repeat(1, pad_len)
            idx = torch.cat([pad, idx], dim=1)
            T = idx.size(1)

        # create contexts
        contexts = torch.stack([idx[:, i:i+self.ngram] for i in range(T - self.ngram + 1)], dim=1)
        x = self.token_embedding(contexts)  # (B, num_ctx, ngram, E)

        # add positional embeddings
        pos_idx = torch.arange(self.ngram, device=idx.device)  # [0,1,...,ngram-1]
        pos_emb = self.pos_embedding(pos_idx)                 # (ngram, E)
        x = x + pos_emb                                      # broadcasting

        B, num_ctx, n, E = x.shape
        x = x.view(B, num_ctx, n * E)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        loss = None
        if targets is not None:
            if targets.size(1) < self.ngram:
                pad_len = self.ngram - targets.size(1)
                pad = targets[:, :1].repeat(1, pad_len)
                targets = torch.cat([pad, targets], dim=1)
            targets_shifted = targets[:, self.ngram - 1 : self.ngram - 1 + num_ctx]
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   targets_shifted.reshape(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            # --- Handle short starting sequences ---
            if idx.size(1) < self.ngram:
                pad_len = self.ngram - idx.size(1)
                pad = idx[:, :1].repeat(1, pad_len)
                context = torch.cat([pad, idx], dim=1)
            else:
                context = idx[:, -self.ngram:]

            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            # --- Top-k filtering ---
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(1)
                logits[logits < min_v] = -float('inf')

            # --- Sample next token ---
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx