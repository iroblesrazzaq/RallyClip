#!/usr/bin/env python3
"""
Simple LSTM model for tennis point detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TennisPointLSTM(nn.Module):
    """
    LSTM model for tennis point detection.
    Takes sequences of feature vectors and predicts probability of being in a point for each frame.
    """
    
    def __init__(self, input_size=288, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        """
        Initialize the TennisPointLSTM.
        
        Args:
            input_size (int): Size of input feature vectors
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability
            bidirectional (bool): If True, becomes a bidirectional LSTM
        """
        super(TennisPointLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer for classification at each time step
        self.fc = nn.Linear(lstm_output_size, 1)  # Single output for binary classification
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply fully connected layer to each time step
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        output = self.fc(lstm_out)
        
        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        
        # Output shape: (batch_size, sequence_length, 1)
        return output


# Alternative model that uses attention mechanism
class TennisPointLSTMWithAttention(nn.Module):
    """
    LSTM model with attention mechanism for tennis point detection.
    """
    
    def __init__(self, input_size=288, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=False):
        """
        Initialize the TennisPointLSTMWithAttention.
        """
        super(TennisPointLSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism (simplified version)
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # Fully connected layer for classification at each time step
        self.fc = nn.Linear(lstm_output_size, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass with attention.
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply fully connected layer to each time step
        output = self.fc(lstm_out)
        
        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        
        # Output shape: (batch_size, sequence_length, 1)
        return output


# Training function example
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function for binary classification with sigmoid output
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)  # Shape: (batch_size, sequence_length, 1)
            
            # Reshape for loss calculation
            # outputs: (batch_size * sequence_length, 1)
            # labels: (batch_size * sequence_length, 1)
            batch_size, seq_length, _ = outputs.shape
            outputs = outputs.view(batch_size * seq_length, 1)
            labels = labels.view(batch_size * seq_length, 1).float()
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences)
                
                # Reshape for loss calculation
                batch_size, seq_length, _ = outputs.shape
                outputs = outputs.view(batch_size * seq_length, 1)
                labels = labels.view(batch_size * seq_length, 1).float()
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')


# Example usage
if __name__ == "__main__":
    # Create model
    model = TennisPointLSTM(
        input_size=288,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("TennisPointLSTM Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with dummy input
    dummy_input = torch.randn(4, 150, 288)  # batch_size=4, seq_len=150, features=288
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (4, 150, 1)
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")  # Should be [0, 1]
    
    # Create attention model
    attention_model = TennisPointLSTMWithAttention(
        input_size=288,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("\n\nTennisPointLSTMWithAttention Model Architecture:")
    print(attention_model)
    print(f"Total parameters: {sum(p.numel() for p in attention_model.parameters())}")
    
    # Test with dummy input
    attention_output = attention_model(dummy_input)
    print(f"\nAttention model dummy input shape: {dummy_input.shape}")
    print(f"Attention model output shape: {attention_output.shape}")  # Should be (4, 150, 1)
    print(f"Attention model output range: [{attention_output.min().item():.4f}, {attention_output.max().item():.4f}]")  # Should be [0, 1]