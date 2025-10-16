# train.py
import torch
import torch.nn as nn
import logging
import random
import numpy as np


# Relative imports from the project structure
from .model import BigramLanguageModel
from .model import MLPBigramLanguageModel
from .model import MLPNgramLanguageModel, MLPNgramLanguageModel_posemd
from . import config

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ],
)


logging.info("Starting new training run...")


logging.info(f"Using device: {config.DEVICE}")
logging.info(f"Log file located at: {config.LOG_FILE}")


# -----------------------------
# Part 1: Data Preparation
# -----------------------------
def get_source_data(full_dataset: bool, sample_size: int):
    """Loads dataset, builds vocab, returns get_batch() closure."""

    with open(config.DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    if not full_dataset:
        data = data[:sample_size]

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
        x = torch.stack([source[i:i+config.BLOCK_SIZE] for i in ix])
        y = torch.stack([source[i+1:i+config.BLOCK_SIZE+1] for i in ix])
        return x.to(config.DEVICE), y.to(config.DEVICE)

    return get_batch, vocab_size, stoi, itos


# -----------------------------
# Part 2: Training Loop
# -----------------------------
def train_loop(model, optimizer, loss_fn, num_epochs, get_batch, stoi, itos):
    """Core training loop with early stopping on validation accuracy."""
    best_test_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        xb, yb = get_batch("train")
        xb = xb.to(config.DEVICE)
        yb = yb.to(config.DEVICE)
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch("val")
            xb = xb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)

            logits, val_loss = model(xb, yb)

            # Align targets exactly with logits length
            preds = torch.argmax(logits, dim=-1)
            targets_shifted = yb[:, -logits.shape[1]:]  # this ensures shapes match

            correct = (preds == targets_shifted).sum().item()
            total = targets_shifted.numel()
            val_accuracy = correct / total


        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ---- Early Stopping ----
        if val_accuracy > best_test_accuracy:
            best_test_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(),"stoi": stoi,"itos": itos}, config.MODEL_SAVE_PATH)

            logging.info(
                f"  -> New best model saved with accuracy: {best_test_accuracy:.4f} "
                f"to {config.MODEL_SAVE_PATH}"
            )
        else:
            epochs_without_improvement += 1
            logging.info(
                f"  -> No improvement. Patience: {epochs_without_improvement}/{config.PATIENCE}"
            )

        if epochs_without_improvement >= config.PATIENCE:
            logging.info(
                f"⏹ Early stopping triggered after {config.PATIENCE} epochs without improvement."
            )
            break

    logging.info(f"🏁 Training finished. Best validation accuracy: {best_test_accuracy:.4f}")


# -----------------------------
# Part 3: Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Part 4: Main Entry Point
# -----------------------------
def main(full_dataset: bool, num_epochs: int, sample_size: int, seed: int = 42):
    set_seed(seed)

    logging.info("🚀 Starting training engine")
    logging.info(f"   - Using full dataset: {full_dataset}")
    if not full_dataset:
        logging.info(f"   - Sample size: {sample_size}")
    logging.info(f"   - Epochs: {num_epochs}")
    logging.info(f"   - Random seed: {seed}")

    # --- Load data ---
    get_batch, vocab_size, stoi, itos = get_source_data(full_dataset, sample_size)

    #model = BigramLanguageModel(vocab_size).to(config.DEVICE)
    if config.MODEL_TYPE == "bigram":
      model = BigramLanguageModel(vocab_size).to(config.DEVICE)
      logging.info("Using BigramLanguageModel")
    elif config.MODEL_TYPE == "mlp":
      model = MLPBigramLanguageModel(vocab_size).to(config.DEVICE)
      logging.info("Using MLPBigramLanguageModel")   
    elif config.MODEL_TYPE == "mlpngram":
      model = MLPNgramLanguageModel(vocab_size).to(config.DEVICE)
    elif config.MODEL_TYPE == "mlpngram_pos":
      model = MLPNgramLanguageModel_posemd(vocab_size).to(config.DEVICE)
      logging.info("Using MLPNgramLanguageModel_posemd")         
    else:
      raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")

    
    logging.info(f"   - Model type: {config.MODEL_TYPE}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    logging.info("Starting training...")
    train_loop(model, optimizer, loss_fn, num_epochs, get_batch, stoi, itos)


if __name__ == "__main__":
    main(full_dataset=True, num_epochs=500, sample_size=5000, seed=42)
