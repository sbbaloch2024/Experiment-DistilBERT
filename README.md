# Experiment-DistilBERT

# DistilBERT for Text Classification on CoLA (GLUE Benchmark)

This project fine-tunes the **DistilBERT** transformer model for sequence classification using the **CoLA dataset** from the GLUE benchmark. The CoLA dataset is designed for linguistic acceptability classification.

## 🚀 Project Overview

- Loads the **CoLA** dataset using Hugging Face's `datasets` library.
- Tokenizes the dataset using the **DistilBERT tokenizer**.
- Fine-tunes **DistilBERT** (`distilbert-base-uncased`) for **binary classification**.
- Uses Hugging Face's `Trainer` API for training and evaluation.
- Saves the **fine-tuned model** for future inference.

## 🛠 Installation

Ensure you have Python 3.7+ and install the required dependencies:

```bash
pip install transformers datasets sklearn numpy torch

##  🛠 Project Structure

│── main.py               # Script for training and evaluating the model
│── README.md             # Project documentation
│── fine_tuned_distilbert/ # Directory for saving the fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── vocab.txt
 
