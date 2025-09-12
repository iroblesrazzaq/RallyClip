"""
Dataset Creator for Tennis Point Detection LSTM Model

This module creates train/validation/test splits from feature-engineered tennis data,
generates overlapping sequences, and saves them in an efficient format for training. 
This is the data pipeline side dataset to save the data, then we have a different 
pytorch inheriting dataset on the model side
"""

import numpy as np
import h5py
import os
import glob
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import shutil

class TennisDatasetCreator:
    """
    Creates train/validation/test datasets from feature-engineered tennis data.
    
    Strategy:
    1. Temporal splitting per video to prevent data leakage
    2. Sequence generation with 50% overlap
    3. Efficient HDF5 storage for large datasets
    """
    
    def __init__(self, feature_dir, output_dir, splits=(0.765, 0.085, 0.15)):
        """
        Initialize dataset creator.
        
        Args:
            feature_dir (str): Directory containing feature NPZ files
            output_dir (str): Directory to save processed datasets
            splits (tuple): (train_ratio, val_ratio, test_ratio)
        """
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio, self.val_ratio, self.test_ratio = splits
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Path to save fitted scaler for inference
        self.scaler_path = self.output_dir / "scaler.joblib"
        
        # Sequence parameters
        self.sequence_length = 150  # 10 seconds at 15 FPS
        self.overlap = 75  # 50% overlap
        self.num_features = 360  # Updated from 288 to match actual feature vectors
        
    def _create_temporal_splits(self, features, targets):
        """
        Split video data temporally to prevent data leakage.
        
        Args:
            features (np.array): Shape (N, num_features)
            targets (np.array): Shape (N,)
            
        Returns:
            dict: Dictionary with 'train', 'val', 'test' splits
        """
        total_frames = len(features)
        
        # Calculate split indices (temporally ordered)
        test_start = int(total_frames * (1 - self.test_ratio))
        val_start = int(total_frames * (1 - self.test_ratio - self.val_ratio))
        
        splits = {
            'train': {
                'features': features[:val_start],
                'targets': targets[:val_start],
                'frame_range': (0, val_start)
            },
            'val': {
                'features': features[val_start:test_start],
                'targets': targets[val_start:test_start],
                'frame_range': (val_start, test_start)
            },
            'test': {
                'features': features[test_start:],
                'targets': targets[test_start:],
                'frame_range': (test_start, total_frames)
            }
        }
        
        return splits
    
    def _create_sequences_with_overlap(self, features, targets):
        """
        Create overlapping sequences from temporal splits.
        
        Args:
            features (np.array): Shape (N, num_features)
            targets (np.array): Shape (N,)
            
        Returns:
            tuple: (sequences, sequence_targets) where each is a list of arrays
        """
        sequences = []
        sequence_targets = []
        
        start_idx = 0
        sequence_count = 0
        
        while start_idx + self.sequence_length <= len(features):
            # Extract sequence
            seq_features = features[start_idx:start_idx + self.sequence_length]
            seq_targets = targets[start_idx:start_idx + self.sequence_length]
            
            sequences.append(seq_features)
            sequence_targets.append(seq_targets)
            
            sequence_count += 1
            
            # Move to next sequence
            start_idx += self.overlap
            
            # Handle edge case: if next sequence would extend beyond data
            if start_idx + self.sequence_length > len(features):
                break
        
        print(f"  Generated {sequence_count} sequences from {len(features)} frames")
        
        return sequences, sequence_targets
    
    def _save_split_to_hdf5(self, sequences, targets, split_name):
        """
        Save sequences to HDF5 file.
        
        Args:
            sequences (list): List of sequence arrays
            targets (list): List of target arrays
            split_name (str): Name of split ('train', 'val', 'test')
        """
        if not sequences:
            print(f"  No sequences to save for {split_name}")
            return
            
        # Convert to numpy arrays
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        # Save to HDF5
        output_file = self.output_dir / f"{split_name}.h5"
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=sequences_array, compression='gzip')
            f.create_dataset('targets', data=targets_array, compression='gzip')
            
        print(f"  Saved {len(sequences)} sequences to {output_file}")
        print(f"    Features shape: {sequences_array.shape}")
        print(f"    Targets shape: {targets_array.shape}")
    
    def process_single_video(self, npz_file):
        """
        Process a single video's feature file.
        
        Args:
            npz_file (str): Path to feature NPZ file
            
        Returns:
            dict: Processing statistics
        """
        print(f"Processing {os.path.basename(npz_file)}")
        
        # Load feature data
        try:
            data = np.load(npz_file, allow_pickle=True)
            features = data['features']
            targets = data['targets']
        except Exception as e:
            print(f"  Error loading {npz_file}: {e}")
            return None
            
        print(f"  Loaded {len(features)} feature vectors")
        
        # Create temporal splits
        splits = self._create_temporal_splits(features, targets)
        
        # Process each split
        split_stats = {}
        for split_name, split_data in splits.items():
            split_features = split_data['features']
            split_targets = split_data['targets']
            
            if len(split_features) == 0:
                print(f"  Skipping {split_name} split (no data)")
                split_stats[split_name] = {'sequences': 0, 'frames': 0}
                continue
                
            print(f"  Creating sequences for {split_name} split "
                  f"({len(split_features)} frames)")
            
            # Create sequences
            sequences, sequence_targets = self._create_sequences_with_overlap(
                split_features, split_targets)
            
            split_stats[split_name] = {
                'sequences': len(sequences),
                'frames': len(split_features),
                'frame_range': split_data['frame_range']
            }
            
            # Save sequences (will be combined with other videos later)
            if sequences:
                self._save_individual_split(sequences, sequence_targets, 
                                          split_name, npz_file)
        
        return {
            'video': os.path.basename(npz_file),
            'total_frames': len(features),
            'splits': split_stats
        }
    
    def _save_individual_split(self, sequences, targets, split_name, npz_file):
        """
        Save sequences from a single video to split-specific file.
        
        Args:
            sequences (list): List of sequence arrays
            targets (list): List of target arrays
            split_name (str): Name of split
            npz_file (str): Source NPZ file path
        """
        if not sequences:
            return
            
        # Create split directory
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Get video name
        video_name = Path(npz_file).stem.replace('_features', '')
        
        # Save to individual file
        output_file = split_dir / f"{video_name}.h5"
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=sequences_array, compression='gzip')
            f.create_dataset('targets', data=targets_array, compression='gzip')
    
    def combine_splits(self):
        """
        Combine individual video splits into final dataset files.
        """
        print("Combining individual splits into final datasets...")
        
        for split_name in ['train', 'val', 'test']:
            split_dir = self.output_dir / split_name
            
            if not split_dir.exists():
                print(f"  No {split_name} directory found")
                continue
                
            # Find all HDF5 files for this split
            h5_files = list(split_dir.glob("*.h5"))
            
            if not h5_files:
                print(f"  No files found for {split_name} split")
                continue
                
            print(f"  Combining {len(h5_files)} files for {split_name} split")
            
            # Load and combine all sequences
            all_features = []
            all_targets = []
            
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        features = f['features'][:]
                        targets = f['targets'][:]
                        all_features.append(features)
                        all_targets.append(targets)
                except Exception as e:
                    print(f"    Error loading {h5_file}: {e}")
                    continue
            
            if not all_features:
                print(f"  No data to combine for {split_name} split")
                continue
                
            # Concatenate all sequences
            combined_features = np.concatenate(all_features, axis=0)
            combined_targets = np.concatenate(all_targets, axis=0)
            
            # Save combined dataset
            output_file = self.output_dir / f"{split_name}.h5"
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('features', data=combined_features, compression='gzip')
                f.create_dataset('targets', data=combined_targets, compression='gzip')
            
            print(f"  Saved combined {split_name} dataset:")
            print(f"    Features shape: {combined_features.shape}")
            print(f"    Targets shape: {combined_targets.shape}")
            print(f"    Total sequences: {len(combined_features)}")
    
    def create_dataset(self):
        """
        Main method to create the complete dataset.
        
        Returns:
            dict: Dataset creation statistics
        """
        print("Creating tennis point detection dataset...")
        print(f"Feature directory: {self.feature_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Splits: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        print(f"Sequence length: {self.sequence_length}, overlap: {self.overlap}")
        
        # Find all feature NPZ files
        npz_files = list(self.feature_dir.glob("*_features.npz"))
        
        if not npz_files:
            print("No feature NPZ files found!")
            return None
            
        print(f"Found {len(npz_files)} feature files")
        
        # Process each video
        all_stats = []
        for npz_file in npz_files:
            stats = self.process_single_video(str(npz_file))
            if stats:
                all_stats.append(stats)
        
        # Combine splits
        self.combine_splits()
        
        # Save dataset info
        dataset_info = {
            'splits': {
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio
            },
            'sequence_params': {
                'length': self.sequence_length,
                'overlap': self.overlap
            },
            'video_stats': all_stats,
            'total_videos': len(npz_files)
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset creation complete!")
        print(f"Dataset info saved to {info_file}")
        
        return dataset_info

    def combine_splits(self):
        """
        Combine individual video splits, apply normalization, and save final dataset files.
        """
        print("Combining individual splits and applying normalization...")
        
        scaler = StandardScaler()
        
        # === Step 1: Fit Scaler ONLY on Training Data ===
        print("  Fitting scaler on training data...")
        train_split_dir = self.output_dir / 'train'
        train_h5_files = list(train_split_dir.glob("*.h5"))
        
        if not train_h5_files:
            print("  No training files found to fit scaler. Aborting.")
            return

        # Load all training features to fit the scaler
        all_train_features_list = []
        for h5_file in train_h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Shape: (num_sequences, seq_len, features)
                    features = f['features'][:]
                    all_train_features_list.append(features)
            except Exception as e:
                print(f"    Error loading {h5_file}: {e}")
        
        if not all_train_features_list:
            print("  No training data loaded. Aborting normalization.")
            return

        # Concatenate and reshape for scaler: (total_frames, num_features)
        combined_train_features = np.concatenate(all_train_features_list, axis=0)
        num_train_sequences, seq_len, num_features = combined_train_features.shape
        reshaped_train_features = combined_train_features.reshape(-1, num_features)
        
        # Fit the scaler
        scaler.fit(reshaped_train_features)
        print(f"  Scaler fitted on {reshaped_train_features.shape[0]} training frames.")
        
        # Save the fitted scaler for future use (e.g., in inference)
        joblib.dump(scaler, self.scaler_path)
        print(f"  Scaler saved to {self.scaler_path}")

        # === Step 2: Transform All Splits and Save Final Datasets ===
        for split_name in ['train', 'val', 'test']:
            split_dir = self.output_dir / split_name
            
            if not split_dir.exists():
                print(f"  No {split_name} directory found")
                continue
                
            h5_files = list(split_dir.glob("*.h5"))
            
            if not h5_files:
                print(f"  No files found for {split_name} split")
                continue
                
            print(f"  Transforming and combining {len(h5_files)} files for {split_name} split...")
            
            all_features = []
            all_targets = []
            
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        all_features.append(f['features'][:])
                        all_targets.append(f['targets'][:])
                except Exception as e:
                    print(f"    Error loading {h5_file}: {e}")
            
            if not all_features:
                continue

            combined_features = np.concatenate(all_features, axis=0)
            combined_targets = np.concatenate(all_targets, axis=0)
            
            # Reshape for transformation
            num_seq, seq_len, num_feat = combined_features.shape
            reshaped_features = combined_features.reshape(-1, num_feat)
            
            # Apply the FITTED scaler
            normalized_features = scaler.transform(reshaped_features)
            
            # Reshape back to sequence format
            final_features = normalized_features.reshape(num_seq, seq_len, num_feat)
            
            # Save combined and normalized dataset
            output_file = self.output_dir / f"{split_name}.h5"
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('features', data=final_features, compression='gzip')
                f.create_dataset('targets', data=combined_targets, compression='gzip')
            
            print(f"  Saved combined and normalized {split_name} dataset:")
            print(f"    Features shape: {final_features.shape}")
            print(f"    Total sequences: {len(final_features)}")

        # === Step 3: Cleanup intermediate per-video split files ===
        for split_dir_name in ['train', 'val', 'test']:
            split_dir_path = self.output_dir / split_dir_name
            if split_dir_path.exists() and split_dir_path.is_dir():
                try:
                    shutil.rmtree(split_dir_path)
                    print(f"  Cleaned intermediate split directory: {split_dir_path}")
                except Exception as e:
                    print(f"  Warning: Failed to remove {split_dir_path}: {e}")

def main():
    """CLI entry point for TennisDatasetCreator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create tennis point detection dataset")
    parser.add_argument("feature_dir", help="Directory containing feature NPZ files")
    parser.add_argument("--output-dir", "-o", default="data", help="Output directory (default: data)")
    parser.add_argument("--train-ratio", type=float, default=0.765, help="Training split ratio (default: 0.765)")
    parser.add_argument("--val-ratio", type=float, default=0.085, help="Validation split ratio (default: 0.085)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    
    args = parser.parse_args()
    
    # Validate split ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Split ratios sum to {total_ratio}, not 1.0")
    
    # Initialize creator
    creator = TennisDatasetCreator(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        splits=(args.train_ratio, args.val_ratio, args.test_ratio)
    )
    
    # Create dataset
    stats = creator.create_dataset()
    
    if stats:
        print("\nDataset Statistics:")
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()