"""
Command-line entry point for training the Bigram/Transformer model.

Usage:
  # Quick sample run:
  python train.py --use_sample --num_epochs 5 --sample_size 2000

  # Full dataset run:
  python train.py --num_epochs 20
"""

import argparse
import logging
import random
import numpy as np
import torch

from nanoGPT import training_engine, config

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Bigram LM Training.")
    parser.add_argument("--use_sample", action="store_true", help="Use a sample of the dataset instead of full dataset")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of records if using sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # --- Determine dataset usage ---
    full_dataset = not args.use_sample

    # --- Set random seeds ---
    set_seed(args.seed)

    # --- User Feedback ---
    logging.info("▶ Training run initiated with the following configuration:")
    logging.info(f"   - Full Dataset Used: {full_dataset}")
    if not full_dataset:
        logging.info(f"   - Sample Size:       {args.sample_size}")
    logging.info(f"   - Max Epochs:        {args.num_epochs}")
    logging.info(f"   - Random Seed:       {args.seed}")
    logging.info(f"   - Model Save Path:   {config.MODEL_SAVE_PATH}")
    logging.info("-" * 40)

    # --- Execute Core Logic ---
    training_engine.main(
        full_dataset=full_dataset,
        num_epochs=args.num_epochs,
        sample_size=args.sample_size,
        seed=args.seed
    )

    logging.info("✅ Training run script finished.")
