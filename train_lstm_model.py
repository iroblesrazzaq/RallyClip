#!/usr/bin/env python3
"""
Training script for Tennis Point Detection LSTM Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt

from tennis_dataset import TennisDataset
from lstm_model_arch import TennisPointLSTM

def evaluate_model(model, data_loader, criterion, device, pos_weight=None):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader for the dataset
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for computation
        pos_weight (float): Positive class weight for weighted loss (optional)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            
            # Reshape for loss calculation
            batch_size, seq_length, _ = outputs.shape
            outputs_reshaped = outputs.view(batch_size * seq_length, 1)
            labels_reshaped = labels.view(batch_size * seq_length, 1).float()
            
            # Calculate loss
            if pos_weight is not None and pos_weight > 0:
                # For weighted evaluation, we need to use BCEWithLogitsLoss
                pos_weight_tensor = torch.tensor([pos_weight]).to(device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                # We need raw logits for BCEWithLogitsLoss
                loss = loss_fn(outputs_reshaped, labels_reshaped)
            else:
                loss = criterion(outputs_reshaped, labels_reshaped)
            total_loss += loss.item()
            
            # Calculate accuracy using probabilities
            predictions = (torch.sigmoid(outputs_reshaped) > 0.5).float()
            total_correct += (predictions == labels_reshaped).sum().item()
            total_samples += labels_reshaped.numel()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses over epochs
        val_losses (list): Validation losses over epochs
        train_accuracies (list): Training accuracies over epochs
        val_accuracies (list): Validation accuracies over epochs
        save_path (str): Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_accuracy, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer to save
        epoch (int): Current epoch
        train_loss (float): Current training loss
        val_loss (float): Current validation loss
        val_accuracy (float): Current validation accuracy
        checkpoint_dir (str): Directory to save checkpoints
        is_best (bool): Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")

def train_model(model, train_loader, val_loader, test_loader, 
                num_epochs=50, learning_rate=0.001, batch_size=32,
                pos_weight=None, early_stopping_patience=10, early_stopping_threshold=None,
                checkpoint_dir='checkpoints', log_dir='logs'):
    """
    Train the tennis point detection model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size
        pos_weight (float): Positive class weight for weighted loss
        early_stopping_patience (int): Number of epochs to wait before early stopping
        early_stopping_threshold (float): Validation loss threshold for early stopping (optional)
        checkpoint_dir (str): Directory to save model checkpoints
        log_dir (str): Directory to save logs
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Loss function and optimizer
    if pos_weight is not None and pos_weight > 0:
        # Use weighted BCEWithLogitsLoss
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Using weighted loss with positive weight: {pos_weight}")
    else:
        # Standard BCE loss
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        print("Using standard BCE loss")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_since_improvement = 0
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            
            # Reshape for loss calculation
            batch_size, seq_length, _ = outputs.shape
            outputs_reshaped = outputs.view(batch_size * seq_length, 1)
            labels_reshaped = labels.view(batch_size * seq_length, 1).float()
            
            # Calculate loss
            if pos_weight is not None and pos_weight > 0:
                # For weighted loss, we use raw logits
                loss = criterion(outputs_reshaped, labels_reshaped)
            else:
                # Standard BCELoss
                loss = criterion(outputs_reshaped, labels_reshaped)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            # Calculate accuracy using probabilities
            predictions = (torch.sigmoid(outputs_reshaped) > 0.5).float()
            train_correct += (predictions == labels_reshaped).sum().item()
            train_total += labels_reshaped.numel()
            
            if batch_idx % 10 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, 
                                              pos_weight if pos_weight is not None and pos_weight > 0 else None)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Check if this is the best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            
        # Early stopping check
        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        # Check early stopping conditions
        should_stop = False
        
        # Patience-based early stopping
        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping: No improvement in validation loss for {early_stopping_patience} epochs")
            should_stop = True
            
        # Threshold-based early stopping
        if early_stopping_threshold is not None and val_loss <= early_stopping_threshold:
            print(f"Early stopping: Validation loss {val_loss:.4f} reached threshold {early_stopping_threshold:.4f}")
            should_stop = True
            
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, avg_train_loss, val_loss, val_accuracy, 
                       checkpoint_dir, is_best)
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
            
        # Early stopping
        if should_stop:
            print(f"Stopping training at epoch {epoch+1}")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device,
                                            pos_weight if pos_weight is not None and pos_weight > 0 else None)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save final test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f)
    
    # Plot training curves
    plot_path = os.path.join(log_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, plot_path)
    
    return history

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train Tennis Point Detection LSTM Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--pos-weight', type=float, default=4.0, 
                        help='Positive class weight for imbalanced dataset (default: 4.0, 0 to disable)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping (0 to disable)')
    parser.add_argument('--early-stopping-threshold', type=float, default=None,
                        help='Validation loss threshold for early stopping (default: None)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    print("=== Tennis Point Detection Model Training ===")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  LSTM Layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Positive Weight: {args.pos_weight}")
    print(f"  Early Stopping Patience: {args.early_stopping_patience}")
    print(f"  Early Stopping Threshold: {args.early_stopping_threshold}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print(f"  Log Dir: {args.log_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = TennisDataset('data/train.h5')
    val_dataset = TennisDataset('data/val.h5')
    test_dataset = TennisDataset('data/test.h5')
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model - use return_logits=True when using weighted loss
    print("\nCreating model...")
    model = TennisPointLSTM(
        input_size=360,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        return_logits=(args.pos_weight is not None and args.pos_weight > 0)  # Return logits for weighted loss
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight if args.pos_weight > 0 else None,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()