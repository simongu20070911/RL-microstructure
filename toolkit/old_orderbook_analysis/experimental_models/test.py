import torch
import time
import random
import numpy as np
# models.py
import torch
import torch.nn as nn
import random
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, n_classes=3, dropout=0.2):
        super(TransformerModel, self).__init__()
        """
        A minimal Transformer Encoder for sequence classification with dropout to prevent overfitting.
        - input_dim : number of features
        - d_model   : embedding dimension for the transformer
        - nhead     : number of attention heads
        - num_layers: number of transformer encoder layers
        - n_classes : 3 for up, down, flat
        - dropout   : dropout probability
        """
        self.d_model = d_model
        
        # Linear embedding of input features to d_model
        self.feature_embed = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout layer for feature embedding
        
        # Positional Encoding (simple learnable or sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model))  # 1000 max sequence length, can be bigger

        # Transformer Encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with dropout
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),  # Dropout before classification
            nn.Linear(d_model, n_classes)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1) Project features to d_model
        x = self.feature_embed(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)  # Apply dropout after embedding
        
        # 2) Add positional encoding (basic approach: just slice the first seq_len positions)
        # shape of pos_embed: (1, max_seq_len, d_model)
        pos = self.pos_embed[:, :seq_len, :]
        x = x + pos  # broadcast add: (batch_size, seq_len, d_model)
        
        # 3) Transformer encoder expects (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  # -> (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)  # -> (seq_len, batch_size, d_model)
        
        # 4) Take the last time step (or pool)
        x = x[-1, :, :]  # (batch_size, d_model)
        
        # 5) Classification
        out = self.fc_out(x)  # (batch_size, n_classes)
        
        return out
class EnsembleModel(nn.Module):
    def __init__(self, num_models, input_dim, d_model=64, nhead=4, num_layers=2, n_classes=3, dropout=0.1):
        super(EnsembleModel, self).__init__()
        """
        Ensemble of Transformer models.
        - num_models: Number of Transformer models in the ensemble
        - input_dim : Number of features
        - d_model   : Embedding dimension for each Transformer
        - nhead     : Number of attention heads
        - num_layers: Number of Transformer encoder layers
        - n_classes : Number of classes (for classification)
        - dropout   : Dropout probability
        """
        self.models = nn.ModuleList([
            TransformerModel(input_dim, d_model, nhead, num_layers, n_classes, dropout) 
            for _ in range(num_models)
        ])
    
    def forward(self, x):
        """
        Aggregate outputs from all models in the ensemble.
        x: (batch_size, seq_len, input_dim)
        Returns:
        - Averaged logits across all models: (batch_size, n_classes)
        """
        outputs = [model(x) for model in self.models]  # List of outputs from each model
        outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, n_classes)
        out = outputs.mean(dim=0)  # Average logits across models
        return out
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=3):
        super(LSTMModel, self).__init__()
        """
        A simple LSTM for classification.
        """
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # out shape: (batch_size, seq_len, hidden_dim)
        # Let's take the last hidden state for classification
        last_out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(last_out)   # (batch_size, n_classes)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=3):
        super(GRUModel, self).__init__()
        """
        A simple GRU for classification.
        """
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, h_n = self.gru(x)
        # out shape: (batch_size, seq_len, hidden_dim)
        last_out = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(last_out)   # (batch_size, n_classes)
        return out


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=3, kernel_size=3):
        super(CNNLSTMModel, self).__init__()
        """
        A simple CNN + LSTM hybrid.
        We'll do a small 1D conv over time dimension, then feed into LSTM.
        """
        self.hidden_dim = hidden_dim
        
        # 1D Conv: in_channels=input_dim, out_channels=input_dim (or different), kernel_size
        # But PyTorch expects (N, C, L). We'll treat features as "channels" or we treat "channels=1"?
        # We'll assume each feature is a separate channel => we need to rearrange data.
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        
        # Then LSTM: input_dim = hidden_dim (from conv), hidden_dim, etc
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # We want (batch_size, in_channels, seq_len) for Conv1d
        x = x.permute(0, 2, 1)  # -> (batch_size, input_dim, seq_len)
        
        x = self.conv1(x)       # -> (batch_size, hidden_dim, seq_len)
        x = nn.functional.relu(x)
        
        # Now permute back for LSTM
        x = x.permute(0, 2, 1)  # -> (batch_size, seq_len, hidden_dim)
        
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]  
        out = self.fc(last_out)
        return out
class LSTM_Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=3):
        super(LSTM_Attn, self).__init__()
        """
        LSTM with attention mechanism for sequence classification.
        - input_dim : Number of input features
        - hidden_dim: LSTM hidden state size
        - num_layers: Number of LSTM layers
        - n_classes : Number of output classes
        """
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attention_energy = self.attention(lstm_out).squeeze(2)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_energy, dim=1)  # (batch_size, seq_len)
        
        # Compute context vector as weighted sum of lstm_out
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, hidden_dim)
        
        # Classify
        out = self.fc(context)
        return out
# models.py
class MLPModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dims=None, num_layers=3, 
                 start_dim=256, decay_factor=2, n_classes=3, dropout=0.2):
        super(MLPModel, self).__init__()
        """
        Deep MLP with flexible depth control
        - Choose EITHER:
          a) hidden_dims: list of specific layer dimensions [256, 128, 64] (direct control)
          b) num_layers: 3 with start_dim=256, decay_factor=2 → [256, 128, 64] (auto-dim)
        """
        self.flatten = nn.Flatten()
        input_features = input_dim * seq_len

        # Auto-generate hidden_dims if not provided
        if hidden_dims is None:
            hidden_dims = [start_dim // (decay_factor ** i) 
                          for i in range(num_layers)]
            
        layers = []
        prev_dim = input_features
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
    

class TransformerEncoderOnly(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, n_classes=3, dropout=0.2):
        super(TransformerEncoderOnly, self).__init__()
        """
        Simplified Transformer Encoder for classification without positional encoding.
        Uses average pooling across sequence length instead of last token.
        - input_dim : number of input features
        - d_model   : embedding dimension
        - nhead     : number of attention heads
        - num_layers: number of encoder layers
        - n_classes : number of output classes
        - dropout   : dropout probability
        """
        self.d_model = d_model
        
        # Feature embedding
        self.feature_embed = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Embed features
        x = self.feature_embed(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # Reformat for transformer (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        
        # Average pooling across sequence dimension
        x = x.mean(dim=0)  # (batch_size, d_model)
        
        # Final classification
        out = self.fc_out(x)
        
        return out
    
class EnsemblerModel(nn.Module):
    def __init__(
        self, 
        num_models, 
        input_dim, 
        d_model=64, 
        nhead=4, 
        num_layers=2, 
        n_classes=3, 
        dropout=0.1, 
        seeds=None  # Optional: Provide a list of seeds
    ):
        super(EnsemblerModel, self).__init__()
        if seeds is None:
            seeds = [random.randint(0, 10**6) for _ in range(num_models)]
        assert len(seeds) == num_models, "Number of seeds must match num_models"
        
        self.models = nn.ModuleList()
        for seed in seeds:
            # Isolate RNG state during initialization
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = TransformerModel(
                    input_dim, d_model, nhead, num_layers, n_classes, dropout
                )
                self.models.append(model)

    def forward(self, x, return_variance=False):
        """Returns mean (and optionally variance) of ensemble predictions."""
        outputs = torch.stack([model(x) for model in self.models], dim=0)  # [num_models, batch, n_classes]
        mean = outputs.mean(dim=0)
        if return_variance:
            variance = outputs.var(dim=0)  # Variance across models
            return mean, variance
        return mean

class LSTMEnsemble(nn.Module):
    def __init__(
        self, 
        num_models, 
        input_dim, 
        hidden_dim=64, 
        num_layers=1, 
        n_classes=3, 
        seeds=None  # Optional: Provide a list of seeds
    ):
        super(LSTMEnsemble, self).__init__()
        if seeds is None:
            seeds = [random.randint(0, 10**6) for _ in range(num_models)]
        assert len(seeds) == num_models, "Number of seeds must match num_models"
        
        self.models = nn.ModuleList()
        for seed in seeds:
            # Isolate RNG state during initialization
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    n_classes=n_classes
                )
                self.models.append(model)

    def forward(self, x, return_variance=False):
        """Returns mean (and optionally variance) of ensemble predictions."""
        outputs = torch.stack([model(x) for model in self.models], dim=0)  # [num_models, batch, n_classes]
        mean = outputs.mean(dim=0)
        if return_variance:
            variance = outputs.var(dim=0)  # Variance across models
            return mean, variance
        return mean


class HybridEnsemble(nn.Module):
    def __init__(
        self, 
        num_transformers, 
        num_lstms,
        input_dim, 
        transformer_d_model=64, 
        transformer_nhead=4, 
        transformer_num_layers=1,
        lstm_hidden_dim=64, 
        lstm_num_layers=1,
        n_classes=3, 
        dropout=0.1,
        transformer_seeds=None,
        lstm_seeds=None
    ):
        super(HybridEnsemble, self).__init__()
        # Handle seeds
        if transformer_seeds is None:
            transformer_seeds = [random.randint(0, 10**6) for _ in range(num_transformers)]
        if lstm_seeds is None:
            lstm_seeds = [random.randint(0, 10**6) for _ in range(num_lstms)]
            
        assert len(transformer_seeds) == num_transformers, "Transformer seeds mismatch"
        assert len(lstm_seeds) == num_lstms, "LSTM seeds mismatch"
        
        self.models = nn.ModuleList()
        
        # Initialize Transformers
        for seed in transformer_seeds:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = TransformerModel(
                    input_dim=input_dim,
                    d_model=transformer_d_model,
                    nhead=transformer_nhead,
                    num_layers=transformer_num_layers,
                    n_classes=n_classes,
                    dropout=dropout
                )
                self.models.append(model)
                
        # Initialize LSTMs
        for seed in lstm_seeds:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=lstm_hidden_dim,
                    num_layers=lstm_num_layers,
                    n_classes=n_classes
                )
                self.models.append(model)

    def forward(self, x, return_variance=False):
        """Combine predictions from both Transformer and LSTM models."""
        outputs = torch.stack([model(x) for model in self.models], dim=0)  # [total_models, batch, n_classes]
        mean = outputs.mean(dim=0)
        if return_variance:
            variance = outputs.var(dim=0)
            return mean, variance
        return mean
# Assuming all model classes (TransformerModel, LSTMModel, etc.) have been defined in the 'models.py' file

# Function to generate a random dataset with the specified dimensions
def generate_data(input_size, seq_len, batch_size=1):
    # Random data with batch_size, seq_len, input_size
    return torch.randn(batch_size, seq_len, input_size)

# Function to benchmark a model
def benchmark_model(model, input_data, device):
    # Move the model and input data to the specified device
    model = model.to(device)
    input_data = input_data.to(device)
    
    # Start the timer
    start_time = time.time()
    
    # Perform a forward pass (no gradients needed for benchmarking)
    with torch.no_grad():
        output = model(input_data)
    
    # Measure the time taken
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return elapsed_time_ms

# Function to initialize the model (you can pass different models and hyperparameters)
def get_model(model_name, input_size, output_size, d_model=64, nhead=4, num_layers=3, dropout=0.1, trans_lstm_hidden=64, trans_lstm_layers=1):
    if model_name == "Transformer":
        return TransformerModel(input_size, d_model, nhead, num_layers, output_size, dropout)
    elif model_name == "LSTM":
        return LSTMModel(input_size, trans_lstm_hidden, trans_lstm_layers, output_size)
    elif model_name == "GRU":
        return GRUModel(input_size, trans_lstm_hidden, trans_lstm_layers, output_size)
    elif model_name == "MLP":
        return MLPModel(input_size, seq_len=L, n_classes=output_size, dropout=dropout)
    elif model_name == "CNN_LSTM":
        return CNNLSTMModel(input_size, trans_lstm_hidden, trans_lstm_layers, output_size)
    elif model_name == "LSTM_Attn":
        return LSTM_Attn(input_size, trans_lstm_hidden, trans_lstm_layers, output_size)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized")

# Function to run benchmarks for different models
def run_benchmarks(input_size, seq_len, batch_size=1, output_size=3, device="cpu"):
    models = ["Transformer", "LSTM", "GRU", "MLP", "CNN_LSTM", "LSTM_Attn"]
    
    # Generate random input data
    input_data = generate_data(input_size, seq_len, batch_size)
    
    # Dictionary to hold the benchmark results
    benchmark_results = {}
    
    for model_name in models:
        print(f"Benchmarking {model_name} model...")
        model = get_model(model_name, input_size, output_size)
        
        # Benchmark the model
        elapsed_time_ms = benchmark_model(model, input_data, device)
        benchmark_results[model_name] = elapsed_time_ms
        print(f"{model_name} model time: {elapsed_time_ms:.4f} ms")
    
    return benchmark_results

# Set the parameters for benchmarking
input_size = 64  # Features
L = 100  # Sequence length
batch_size = 1  # Batch size
output_size = 3  # Number of output classes (for classification)
device = "cpu"  # CPU device (you can switch to "cuda" if using GPU)

# Run the benchmarks
benchmark_results = run_benchmarks(input_size, L, batch_size, output_size, device)

# Print the final results
print("\nBenchmark results (in ms):")
for model_name, time_taken in benchmark_results.items():
    print(f"{model_name}: {time_taken:.4f} ms")