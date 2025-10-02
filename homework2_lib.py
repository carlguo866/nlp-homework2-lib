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

def calculate_perplexity(model, tokenizer, test_prompts: List[str], batch_size: int = 16, max_length: int = 512):
    """
    Calculate perplexity of model with test prompts in a batched way.
    Applies attention mask so that only valid tokens contribute to the loss.
    """
    model.eval()
    device = next(model.parameters()).device

    print("Calculating perplexity of model with test prompts (batched):")
    print("=" * 60)

    # Tokenize all prompts at once (batched)
    encodings = tokenizer(
        test_prompts,
        max_length=max_length,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    num_samples = input_ids.size(0)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size)):
            end = min(start + batch_size, num_samples)
            batch_input_ids = input_ids[start:end]

            # Forward pass
            logits = model(batch_input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_input_ids[..., 1:].contiguous()
            if attention_mask is not None:
                batch_attention_mask = attention_mask[start:end]
                shift_mask = batch_attention_mask[..., 1:].contiguous()
            else:
                shift_mask = None

            # Flatten for loss computation
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )  # (batch * seq_len-1,)

            if shift_mask is not None:
                loss = loss * shift_mask.view(-1).float()
                num_valid = shift_mask.sum().item()
            else:
                num_valid = shift_labels.numel()

            total_loss += loss.sum().item()
            total_tokens += num_valid

    avg_loss = total_loss / max(1, total_tokens)
    avg_perplexity = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    return avg_perplexity



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