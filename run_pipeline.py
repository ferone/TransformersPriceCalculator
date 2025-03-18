"""
Transformer Price Calculator - Model Training Pipeline

This script runs the complete pipeline for:
1. Generating synthetic data with realistic weight scaling
2. Analyzing weight-power relationships
3. Training and evaluating models
4. Saving model results and visualizations
"""

import os
import sys
import argparse
import time
from pathlib import Path

def create_directories():
    """Create necessary directories for the pipeline"""
    directories = [
        'data',
        'models',
        'visualizations',
        'visualizations/weight_analysis'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_pipeline(args):
    """Run the complete model training pipeline"""
    start_time = time.time()
    
    # Create necessary directories
    create_directories()
    
    # Generate synthetic data with realistic weight scaling
    if not args.skip_data_generation:
        print("\n" + "="*50)
        print("Step 1: Generating synthetic transformer data with realistic weight scaling")
        print("="*50)
        
        try:
            from src.data_generation import generate_synthetic_data, verify_weight_scaling
            
            # Generate data
            n_samples = args.samples if args.samples else 1000
            transformer_data = generate_synthetic_data(n_samples)
            
            # Verify weight scaling
            weight_scaling = verify_weight_scaling(transformer_data)
            
            # Save data
            transformer_data.to_csv('data/transformer_data.csv', index=False)
            weight_scaling.to_csv('data/weight_scaling.csv', index=False)
            
            print(f"Generated {len(transformer_data)} transformer data samples with realistic weight scaling")
        except Exception as e:
            print(f"Error generating data: {str(e)}")
            if not args.continue_on_error:
                sys.exit(1)
    else:
        print("Skipping data generation...")
    
    # Analyze weight scaling and enhance dataset with weight-related features
    if not args.skip_analysis:
        print("\n" + "="*50)
        print("Step 2: Analyzing weight scaling relationships")
        print("="*50)
        
        try:
            from src.model_training import analyze_weight_scaling, add_weight_features
            from src.data_processing import load_data
            
            # Load data
            df = load_data('data/transformer_data.csv')
            
            # Analyze weight scaling
            df = analyze_weight_scaling(df)
            
            # Add weight-related features
            df = add_weight_features(df)
            
            # Save enhanced dataset
            df.to_csv('data/transformer_data_enhanced.csv', index=False)
            print(f"Enhanced dataset saved with {df.shape[1]} features")
        except Exception as e:
            print(f"Error analyzing weight scaling: {str(e)}")
            if not args.continue_on_error:
                sys.exit(1)
    else:
        print("Skipping weight scaling analysis...")
    
    # Train and evaluate models
    if not args.skip_training:
        print("\n" + "="*50)
        print("Step 3: Training and evaluating models")
        print("="*50)
        
        try:
            # Import here to ensure modifications to src.model_training are picked up
            # even if this script was already imported
            from importlib import reload
            import src.model_training
            reload(src.model_training)
            from src.model_training import main as train_models
            
            # Train models
            train_models()
        except Exception as e:
            print(f"Error training models: {str(e)}")
            if not args.continue_on_error:
                sys.exit(1)
    else:
        print("Skipping model training...")
    
    # Print execution time
    end_time = time.time()
    print(f"\nTotal pipeline execution time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the transformer price prediction pipeline with weight scaling')
    parser.add_argument('--skip-data-generation', action='store_true', help='Skip synthetic data generation')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip weight scaling analysis')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue execution if a step fails')
    parser.add_argument('--samples', type=int, help='Number of samples to generate (default: 1000)')
    
    args = parser.parse_args()
    
    run_pipeline(args) 