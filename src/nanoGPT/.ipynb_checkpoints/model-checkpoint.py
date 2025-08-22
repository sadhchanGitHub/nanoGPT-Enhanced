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
