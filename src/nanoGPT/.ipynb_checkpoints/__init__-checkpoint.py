"""
Initializes the `nanoGPT` package.

This package contains all the source code required to build,
train, and run the Bigram/Transformer-based models.

By importing the modules here, they can be accessed conveniently:
e.g., `nanoGPT.model.BigramLanguageModel`.

Attributes:
    config (module): Hyperparameters and configurations.
    model (module): BigramLanguageModel architecture.
    training_engine (module): Training logic for the model.
    generate_engine (module): Generation logic for sampling text.
"""

from . import config
from . import model


# Public API
__all__ = ["config", "model"]
