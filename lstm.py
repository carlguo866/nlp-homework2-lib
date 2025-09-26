#!/usr/bin/env python
# coding: utf-8

#%% [markdown]
# # LSTM Training from Scratch
# 
# This script trains an LSTM language model using PyTorch's built-in LSTM implementation.
# We'll use the same TinyStories dataset for consistency with the transformer homework.

#%% Setup and Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer
import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math

#%% [markdown]
# ## LSTM Language Model Implementation

#%% LSTM Language Model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int):
        """
        LSTM Language Model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden state size of LSTM
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target labels for loss computation (batch_size, seq_len)
            hidden: Previous hidden state (h_0, c_0)
            
        Returns:
            If labels provided: (loss, logits)
            Else: logits
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Output projection
        logits = self.output_projection(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 for padding
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, logits
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text using the LSTM language model
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize hidden state
        hidden = self.init_hidden(batch_size, device)
        
        # Generate tokens autoregressively
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits, hidden = self.forward(input_ids, labels=None, hidden=hidden)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

#%% Training Configuration
@dataclass
class LSTMConfig:
    """LSTM model configuration"""
    # Model hyperparameters
    vocab_size: int = 50257
    embedding_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 5
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Paths
    output_dir: str = "./lstm_model"
    log_dir: str = "./lstm_logs"
