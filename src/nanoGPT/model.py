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


class MLPTrigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size                # ← add this
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)  # learned embeddings
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)  # for trigram, concat 2 previous embeddings
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
  
    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T < 2:
            raise ValueError("Sequence length must be at least 2 for trigram model")

        # Prepare trigram context
        contexts = torch.stack([idx[:, i:i+2] for i in range(T-1)], dim=1)  # (B, T-1, 2)
        x = self.token_embedding(contexts)                                    # (B, T-1, 2, embed_dim)
        x = x.view(B, T-1, -1)                                                # (B, T-1, 2*embed_dim)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                                                  # (B, T-1, vocab_size)

        loss = None
        """
        if targets is not None:
            # Shift targets by one to match the trigram predictions
            targets_shifted = targets[:, 1:]                                   # (B, T-1)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets_shifted.view(-1))
        """
        if targets is not None:
          targets_shifted = targets[:, 1:]           # (B, T-1)
          loss = F.cross_entropy(
              logits.reshape(-1, self.vocab_size),  # <- use reshape
              targets_shifted.reshape(-1)
          )

        return logits, loss
    """
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context = idx[:, -2:]              # last 2 tokens
            logits, _ = self(context)         # (B, 1, vocab_size) or (B, vocab_size)
            logits = logits[:, -1, :]         # pick last step
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
    """
    #✅ This makes the sampling less random and more likely to repeat meaningful sequences.
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """
        idx: (B, T) starting context
        temperature: float <1 for more deterministic output
        top_k: if set, restrict sampling to top-k probable tokens
        """
        for _ in range(max_new_tokens):
            context = idx[:, -2:]               # last 2 tokens
            logits, _ = self(context)           # (B, 1, vocab_size)
            logits = logits[:, -1, :]           # pick last step

            # temperature scaling
            logits = logits / temperature

            if top_k is not None:
                # Keep only top_k logits
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(1)
                logits[logits < min_v] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx





