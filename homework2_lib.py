import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from typing import List, Dict, Optional
import json
import os

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Position-wise feed-forward network
        
        Args:
            hidden_size: Model dimension
            intermediate_size: Hidden dimension of FFN
            activation_fn: Activation function ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x



def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainingMetrics:
    """Track training metrics"""
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.step = 0
    
    def update(self, loss: float, lr: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step += 1
    
    def get_avg_loss(self, last_n: int = 100):
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses[-last_n:])


# Custom dataset class for on-the-fly tokenization
class TinyStoriesDataset:
    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples if max_samples is not None else len(dataset)
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # Get raw text
        text = self.dataset[idx]['text']
        
        # Tokenize on the fly
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'  # Return tensors for direct use
        )
        
        # Create labels (same as input_ids for causal language modeling)
        labels = encoded['input_ids'].clone()
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # Remove batch dimension
            'labels': labels.squeeze(0)  # Remove batch dimension
        }


def evaluate_model(model, tokenizer, test_prompts: List[str], temperature: float = 0.7):
    """Evaluate model with test prompts"""
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    print("Generating samples from trained model:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        print("-" * 40)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with different temperatures
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids, 
                max_new_tokens=150,
                temperature=temperature
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Temperature {temperature}: {generated_text}")
            print()

def calculate_perplexity(model, tokenizer, test_prompts: List[str]):
    """Calculate perplexity of model with test prompts"""
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    print("Calculating perplexity of model with test prompts:")
    print("=" * 60)
    
    total_perplexity = 0.0
    num_prompts = len(test_prompts)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}: '{prompt}'")
            print("-" * 40)
            
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Calculate perplexity
            perplexity = calculate_single_perplexity(model, input_ids)
            print(f"Perplexity: {perplexity:.4f}")
            print()
            
            total_perplexity += perplexity

    avg_perplexity = total_perplexity / num_prompts
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    return avg_perplexity

def calculate_single_perplexity(model, input_ids):
    """Calculate perplexity for a single input sequence"""
    model.eval()
    
    with torch.no_grad():
        # Get logits from model
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Perplexity is exp(loss)
        perplexity = torch.exp(loss).item()
        
    return perplexity


class TestDataLoader:
    def __init__(self, test_data_dir: str = "test_results"):
        self.test_data_dir = test_data_dir
        
    def load_test_data(self, component_name: str) -> Optional[Dict]:
        """Load test data for a specific component"""
        filename = os.path.join(self.test_data_dir, f"{component_name.lower()}_tests.json")
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Test data file {filename} not found")
            return None


def save_model(model, tokenizer, save_path: str):
    """Save model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    if hasattr(model, 'config'):
        torch.save(model.config.__dict__, os.path.join(save_path, "config.json"))
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path: str):
    """Load model weights"""
    model.load_state_dict(torch.load(os.path.join(load_path, "model.pt")))
    print(f"Model loaded from {load_path}")