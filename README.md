# Disaster Tweet Verification Project

## Overview
This project focuses on authenticating disaster-related tweets using state-of-the-art natural language processing models, including BERT, RoBERTa, and LSTM. The goal is to enhance the accuracy of discerning genuine disaster-related information from false or misleading tweets. The project involves thorough parameter tuning and comparative analysis of these models to derive the most effective configurations.

## Models Used
1. **BERT (Bidirectional Encoder Representations from Transformers)**
   - BERT Base: 12 transformer blocks, 768 hidden units per layer, 12 attention heads.
   - RoBERTa: Similar to BERT Base but removes the Next Sentence Prediction (NSP) objective, focusing on a dynamically masked language model.

2. **LSTM (Long Short-Term Memory)**
   - Configured with an embedding layer, bidirectional LSTM layer, and fully connected layer.
   - Hyperparameters include embedding dimension, learning rate, hidden units, number of layers, dropout rate, batch size, and sequence length.

## Project Structure
- `505_lstm/`: Contains the Python scripts for data preprocessing, model training, and evaluation.
- 'tweet-disaster-nlp(RoBERTa).ipynb'
- 'tweet-disaster-nlp(bert-base).ipynb'
- 'tweet-disaster-nlp(bert-large).ipynb'

## Project Link
https://github.com/Naserien/CS505/tree/main
