import argparse
import torch
import logging
from nanoGPT.model import BigramLanguageModel
from nanoGPT import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_text(start_char: str, max_new_tokens: int):
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)

    # Initialize model
    model = BigramLanguageModel(vocab_size).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Encode start character
    idx = torch.tensor([[stoi[start_char]]], device=config.DEVICE)

    # Generate sequence
    generated_idx = model.generate(idx, max_new_tokens)
    generated_text = "".join([itos[i.item()] for i in generated_idx[0]])
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained Bigram LM")
    parser.add_argument("--start", type=str, default="\n", help="Starting character")
    parser.add_argument("--length", type=int, default=100, help="Number of new tokens")
    args = parser.parse_args()

    logging.info(f"▶ Generating text starting with '{args.start}' for {args.length} tokens")
    output = generate_text(args.start, args.length)
    print("\n--- GENERATED TEXT ---\n")
    print(output)
    print("\n----------------------\n")
    logging.info("✅ Generation finished.")
