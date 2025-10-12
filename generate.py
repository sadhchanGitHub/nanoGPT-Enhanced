import argparse
import logging
import torch

from nanoGPT import config
from nanoGPT.model import BigramLanguageModel, MLPBigramLanguageModel
from nanoGPT.model import MLPNgramLanguageModel

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_text(start_text: str, max_new_tokens: int):
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    # --- Pick model based on config ---
    if config.MODEL_TYPE == "bigram":
        model = BigramLanguageModel(len(stoi))
    elif config.MODEL_TYPE == "mlp":
        model = MLPBigramLanguageModel(len(stoi))  
    elif config.MODEL_TYPE == "mlpngram":
        model = MLPNgramLanguageModel(len(stoi))             
    else:
        raise ValueError(f"❌ Unknown MODEL_TYPE: {config.MODEL_TYPE}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    # Encode whole start text (not just one char)
    try:
        idx = torch.tensor([[stoi[c] for c in start_text]], device=config.DEVICE)
    except KeyError as e:
        raise ValueError(f"❌ Character {e.args[0]} not in vocabulary")

    # Generate sequence
    # generated_idx = model.generate(idx, max_new_tokens)
    
    # Generate sequence with temperature and top-k
    generated_idx = model.generate(
        idx,
        max_new_tokens,
        temperature=0.8,  # less random early in training
        top_k=5           # only pick from top 5 probable chars
    )

    generated_text = "".join([itos[i.item()] for i in generated_idx[0]])
    return generated_text



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained nanoGPT model")
    parser.add_argument("--start", type=str, default="\n", help="Starting text for generation")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")
    args = parser.parse_args()

    logging.info(f"▶ Generating text starting with '{args.start}' for {args.length} tokens")
    output = generate_text(args.start, args.length)
    print("\n=== Generated Text ===\n")
    print(output)
