"""
Predict Transformer Price Using Trained Models

This script loads the trained models and makes predictions for transformer prices
based on user-provided specifications.
"""

import os
import joblib
import numpy as np
import pandas as pd
import argparse

def load_model(model_name):
    """Load a trained model"""
    model_path = f'models/{model_name}.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def load_preprocessor():
    """Load the preprocessor"""
    preprocessor_path = 'models/preprocessor.joblib'
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    return joblib.load(preprocessor_path)

def predict_price(model, transformer_specs, preprocessor=None):
    """
    Predict transformer price based on specifications
    
    Parameters:
    - model: Trained regression model
    - transformer_specs: Dictionary of transformer specifications
    - preprocessor: Optional preprocessor pipeline
    
    Returns:
    - Predicted price
    """
    # Check required fields
    required_fields = ['Power Rating (KVA)', 'Primary Voltage (kV)', 'Secondary Voltage (kV)', 'Phase', 'Type']
    missing_fields = [field for field in required_fields if field not in transformer_specs]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
    # Convert input specs to DataFrame
    input_df = pd.DataFrame([transformer_specs])
    
    # Check for non-numeric values in numeric fields
    numeric_fields = ['Power Rating (KVA)', 'Primary Voltage (kV)', 'Secondary Voltage (kV)']
    for field in numeric_fields:
        if field in input_df.columns:
            try:
                input_df[field] = pd.to_numeric(input_df[field])
            except ValueError as e:
                raise ValueError(f"Field '{field}' contains non-numeric value: {input_df[field].iloc[0]}")
    
    # Apply preprocessor if provided
    if preprocessor is not None:
        try:
            input_processed = preprocessor.transform(input_df)
            # Make prediction (our model was trained on log-transformed prices)
            log_price = model.predict(input_processed)[0]
            # Convert back from log scale
            price = np.expm1(log_price)
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {str(e)}")
    else:
        # Direct prediction without preprocessing
        try:
            price = model.predict(input_df)[0]
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
    
    return price

def get_available_models():
    """Get list of available trained models"""
    if not os.path.exists('models'):
        return []
    
    model_files = [f for f in os.listdir('models') 
                   if f.endswith('.joblib') and not f.startswith('preprocessor')]
    
    return [os.path.splitext(f)[0] for f in model_files]

def format_price(price):
    """Format price with commas and dollar sign"""
    return f"${price:,.2f}"

def main():
    # Define available models
    available_models = get_available_models()
    
    if not available_models:
        print("No trained models found. Please run train_real_data_model.py first.")
        return
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict transformer price')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=available_models,
                        help=f'Model to use for prediction. Available models: {", ".join(available_models)}')
    parser.add_argument('--power', type=float, required=True,
                        help='Power rating in KVA')
    parser.add_argument('--primary-voltage', type=float, default=11.0,
                        help='Primary voltage in kV')
    parser.add_argument('--secondary-voltage', type=float, default=0.433,
                        help='Secondary voltage in kV')
    parser.add_argument('--phase', type=str, default='Three-phase',
                        choices=['Single-phase', 'Three-phase'],
                        help='Phase type (Single-phase or Three-phase)')
    parser.add_argument('--type', type=str, default='Power',
                        choices=['Power', 'Distribution', 'Auto', 'Instrument'],
                        help='Transformer type')
    parser.add_argument('--all-models', action='store_true',
                        help='Use all available models and show all predictions')
    
    args = parser.parse_args()
    
    # Load preprocessor
    try:
        preprocessor = load_preprocessor()
        print("Loaded preprocessor successfully.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        preprocessor = None
    
    # Prepare transformer specifications - using expected column names
    transformer_specs = {
        'Power Rating (KVA)': args.power,
        'Primary Voltage (kV)': args.primary_voltage,
        'Secondary Voltage (kV)': args.secondary_voltage,
        'Phase': args.phase,
        'Type': args.type
    }
    
    print("\nTransformer Specifications:")
    for key, value in transformer_specs.items():
        print(f"  {key}: {value}")
    
    # Make predictions
    if args.all_models:
        print("\nPredictions from all models:")
        for model_name in available_models:
            try:
                model = load_model(model_name)
                predicted_price = predict_price(model, transformer_specs, preprocessor)
                print(f"  {model_name.replace('_', ' ').title()}: {format_price(predicted_price)}")
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
    else:
        try:
            model = load_model(args.model)
            predicted_price = predict_price(model, transformer_specs, preprocessor)
            print(f"\nPredicted price using {args.model.replace('_', ' ').title()} model: {format_price(predicted_price)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 