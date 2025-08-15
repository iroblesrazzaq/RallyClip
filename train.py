#!/usr/bin/env python3
"""
Tennis Action Recognition Model Training Script

This script loads processed tennis video data and trains a multi-class LSTM model
to classify tennis actions into three categories:
- 0 (INACTIVE): Frame outside any point
- 1 (SERVE_MOTION): First 1.5 seconds of each point  
- 2 (RALLY): Remainder of each point after serve motion
"""

import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def load_and_split_data(data_path):
    """
    Load processed tennis data and split into training and validation sets.
    
    Args:
        data_path (str): Path to the .npz file containing X and y arrays
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    print(f"📂 Loading data from {data_path}...")
    
    # Load the processed data
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"✅ Data loaded successfully!")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Labels shape: {y.shape}")
    
    # Analyze label distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: 'INACTIVE', 1: 'SERVE_MOTION', 2: 'RALLY'}
    print("📊 Label distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"   {label} ({label_names.get(label, 'UNKNOWN')}): {count} samples ({percentage:.1f}%)")
    
    # Split data into training and validation sets
    print("🔀 Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Ensure balanced split across classes
    )
    
    print(f"✅ Data split completed!")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Validation samples: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val


def build_model(input_shape, num_classes):
    """
    Build and compile a multi-class LSTM model for tennis action recognition.
    
    Args:
        input_shape (tuple): Shape of input sequences (sequence_length, features)
        num_classes (int): Number of output classes (3 for tennis actions)
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    print(f"🏗️ Building LSTM model...")
    print(f"   - Input shape: {input_shape}")
    print(f"   - Number of classes: {num_classes}")
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First LSTM layer with return_sequences=True to stack multiple LSTMs
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        
        # Second LSTM layer
        LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        
        # Dense layer with ReLU activation
        Dense(32, activation='relu'),
        
        # Dropout for regularization
        Dropout(0.3),
        
        # Output layer with softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # For integer labels (0, 1, 2)
        metrics=['accuracy']
    )
    
    print("✅ Model built and compiled successfully!")
    print("\n📋 Model Summary:")
    model.summary()
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a multi-class LSTM model for tennis action recognition."
    )
    parser.add_argument(
        "--data_path", 
        required=True, 
        help="Path to the input .npz file created by preprocessing.py"
    )
    parser.add_argument(
        "--model_output_path", 
        required=True, 
        help="Path to save the trained model file (e.g., models/rally_classifier.h5)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Training batch size (default: 32)"
    )
    
    args = parser.parse_args()
    
    print("🎾 Tennis Action Recognition Model Training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Model output: {args.model_output_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.model_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    # Load and split data
    X_train, X_val, y_train, y_val = load_and_split_data(args.data_path)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
    num_classes = len(np.unique(y_train))
    model = build_model(input_shape, num_classes)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint to save best model
    checkpoint_path = args.model_output_path.replace('.h5', '_best.h5')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    print("\n🚀 Starting model training...")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Validation samples: {X_val.shape[0]}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    print("\n📊 Final Model Evaluation:")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"   - Training Accuracy: {train_accuracy:.4f}")
    print(f"   - Training Loss: {train_loss:.4f}")
    print(f"   - Validation Accuracy: {val_accuracy:.4f}")
    print(f"   - Validation Loss: {val_loss:.4f}")
    
    # Save the final model
    print(f"\n💾 Saving final model to {args.model_output_path}...")
    model.save(args.model_output_path)
    
    print("✅ Training completed successfully!")
    print(f"   - Final model saved: {args.model_output_path}")
    print(f"   - Best model saved: {checkpoint_path}")
    
    # Print training summary
    print("\n📈 Training Summary:")
    print(f"   - Total epochs completed: {len(history.history['loss'])}")
    print(f"   - Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   - Final validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()