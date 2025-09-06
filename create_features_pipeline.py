#!/usr/bin/env python3
"""
Tennis Feature Engineering Pipeline

This script uses the TennisFeatureEngineer class to create feature vectors from preprocessed data.
"""

import os
import sys
import argparse
import glob
import time
from data_scripts.feature_engineer import FeatureEngineer

def process_all_preprocessed(input_dir, output_dir, overwrite=False):
    """
    Process all preprocessed files in a directory using the TennisFeatureEngineer.
    
    Args:
        input_dir (str): Directory containing preprocessed .npz files
        output_dir (str): Directory to save feature vectors
        overwrite (bool): Whether to overwrite existing files
    """
    # Find all preprocessed .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_dir, "*_preprocessed.npz"))
    
    if not npz_files:
        print(f"❌ No preprocessed .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} preprocessed files to process")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Process each file
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            # Create output file path
            base_name = os.path.splitext(os.path.basename(npz_file))[0]
            base_name = base_name.replace("_preprocessed", "")  # Remove _preprocessed suffix
            output_file = os.path.join(output_dir, f"{base_name}_features.npz")
            
            print(f"Processing: {os.path.basename(npz_file)}")
            if feature_engineer.create_features_from_preprocessed(npz_file, output_file, overwrite):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"\nFeature engineering complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create feature vectors from preprocessed tennis pose data using TennisFeatureEngineer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_features_pipeline.py --input-dir "preprocessed_data" --output-dir "training_features"
    
    python create_features_pipeline.py --input-dir "preprocessed_data" --output-dir "training_features" --overwrite
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing preprocessed .npz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save feature vectors')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing feature files')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory {args.input_dir} does not exist")
        return
    
    print("=== Tennis Feature Engineering Pipeline ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Process all preprocessed files
    process_all_preprocessed(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )
    
    # Calculate and print total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()