# nanoGPT-Enhanced Transaction Text Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A character-level Bigram/Transformer language model for generating synthetic transaction and merchant text. Built with PyTorch.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating Text](#generating-text)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This repository contains a pure PyTorch implementation of a character-level Bigram/Transformer language model. It is designed to generate synthetic transaction logs, including merchant names, payment types, and POS entries.

---

## Key Features

* **Character-Level Language Model**: Learn patterns at the character level for text generation.
* **Early Stopping**: Stop training automatically when validation accuracy stops improving.
* **Reproducible**: Set random seeds for reproducible results.
* **Command-Line Interface**: Train and generate text directly from CLI scripts.
* **Lightweight and Modular**: Easy to extend with more complex Transformer-based models.

---

## Model Architecture

* **Input Embeddings**: Converts characters to dense vectors.
* **Bigram/Transformer Layer**: Learns character-to-character patterns or token interactions.
* **Softmax Output**: Produces probabilities over the vocabulary for next character prediction.
* **Autoregressive Generation**: Can generate new sequences starting from a given character.

---

## Dataset

The model was trained on synthetic transaction text. 

> Note: Small datasets may produce gibberish; larger datasets improve text quality.

---

## Getting Started

### Prerequisites

* Python 3.10 or higher
* PyTorch 2.0+
* numpy, pandas

Install dependencies:

```bash
pip install torch numpy pandas

You will also need to download the required dataset and place it in the `data/` directory.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/nanoGPT-Enhanced.git
    cd nanoGPT-Enhanced
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```


## Usage

The project includes scripts for training the model from scratch and running inference on new text.

### Training the Model

To start training the model, run the `train.py` script. You can customize the training process using command-line arguments.

```sh
python train.py --num_epochs 100
```
Sample dataset:

```sh
python train.py --use_sample --num_epochs 50 --sample_size 5000
```

#Arguments:

1. --use_sample : Train on subset of the dataset

2. --num_epochs : Number of epochs

3. --sample_size : Number of records if using a sample

Early stopping is controlled via config.PATIENCE.

### Generating Text

Once a model is trained and saved, you can use the `generate.py` script to classify a new piece of text.

```sh
python generate.py --start "P" --length 50
```

#Arguments:

1. --start : Starting character

2. --length : Number of tokens to generate


## Results

After training for 100 epochs, stooped after 18 Epochs due to Pateince = 10
The model achieved the following performance on the test set:


Example output:

--- GENERATED TEXT ---

PLFnZN5Hydky1wRgSw&5xByr38CAusZz2oT
BLzuYygZeyd8LN3
...

Train Loss: 4.9673 | Val Loss: 4.8077 | Val Acc: 0.0078


## Project Structure

```
.
├── data/
│   └── transactions.txt
├── logs/
├── models/
│   └── nanoGPT_enhanced.pth
├── nanoGPT/
│   ├── __init__.py
│   ├── model.py
│   ├── training_engine.py
│   └── config.py
├── train.py
├── generate.py
├── README.md
└── requirements.txt

```

## Contributing

Contributions are welcome! If you have suggestions for improving this project, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   The core concepts and architectural patterns implemented here were learned from and inspired by several excellent educational resources, including Jay Alammar's "The Illustrated Transformer" and Andrej Karpathy's "Let's build GPT".
