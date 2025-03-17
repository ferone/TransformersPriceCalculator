import pandas as pd
import numpy as np
import joblib
import os
import argparse
import sys

# Handle import either from src module or direct execution
try:
    from src.data_processing import preprocess_data
except ModuleNotFoundError:
    from data_processing import preprocess_data

def load_model(model_path):
    """
    Load a trained model from file
    """
    return joblib.load(model_path)

def load_preprocessor(preprocessor_path=None):
    """
    Load the preprocessing pipeline
    """
    if preprocessor_path is None:
        # Check different possible paths
        if os.path.exists('models/preprocessor.joblib'):
            preprocessor_path = 'models/preprocessor.joblib'
        else:
            preprocessor_path = '../models/preprocessor.joblib'
            
    return joblib.load(preprocessor_path)

def predict_price(model, preprocessor, transformer_specs):
    """
    Predict transformer price based on specifications
    
    Parameters:
    - model: Trained regression model
    - preprocessor: Fitted preprocessor pipeline
    - transformer_specs: Dictionary of transformer specifications
    
    Returns:
    - Predicted price
    """
    # Convert input specs to DataFrame
    input_df = pd.DataFrame([transformer_specs])
    
    # Apply preprocessing
    input_processed = preprocessor.transform(input_df)
    
    # Make prediction
    predicted_price = model.predict(input_processed)[0]
    
    return predicted_price

def get_user_input():
    """
    Get transformer specifications from user input
    """
    specs = {}
    
    # Power rating
    power_options = [25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    print("Select power rating (kVA):")
    for i, p in enumerate(power_options):
        print(f"{i+1}. {p}")
    power_choice = int(input("Enter choice (1-11): "))
    specs['power_rating'] = power_options[power_choice-1]
    
    # Voltage ratings
    primary_options = [4160, 12470, 13200, 13800, 24940, 34500]
    print("\nSelect primary voltage (V):")
    for i, v in enumerate(primary_options):
        print(f"{i+1}. {v}")
    primary_choice = int(input("Enter choice (1-6): "))
    specs['primary_voltage'] = primary_options[primary_choice-1]
    
    secondary_options = [120, 208, 240, 277, 480, 600]
    print("\nSelect secondary voltage (V):")
    for i, v in enumerate(secondary_options):
        print(f"{i+1}. {v}")
    secondary_choice = int(input("Enter choice (1-6): "))
    specs['secondary_voltage'] = secondary_options[secondary_choice-1]
    
    # Material weights
    print("\nEnter material weights (kg):")
    specs['core_weight'] = float(input("Core weight (100-3000): "))
    specs['copper_weight'] = float(input("Copper weight (50-2000): "))
    specs['insulation_weight'] = float(input("Insulation weight (20-500): "))
    specs['tank_weight'] = float(input("Tank weight (100-2500): "))
    specs['oil_weight'] = float(input("Oil weight (200-4000): "))
    
    # Calculate total weight
    specs['total_weight'] = (specs['core_weight'] + specs['copper_weight'] + 
                           specs['insulation_weight'] + specs['tank_weight'] + 
                           specs['oil_weight'])
    
    # Other specifications
    cooling_options = ['ONAN', 'ONAF', 'OFAF', 'ODAF']
    print("\nSelect cooling type:")
    for i, c in enumerate(cooling_options):
        print(f"{i+1}. {c}")
    cooling_choice = int(input("Enter choice (1-4): "))
    specs['cooling_type'] = cooling_options[cooling_choice-1]
    
    tap_changer = input("\nDoes the transformer have a tap changer? (y/n): ").lower() == 'y'
    specs['tap_changer'] = tap_changer
    
    specs['efficiency'] = float(input("\nEnter efficiency (0.95-0.99): "))
    specs['impedance'] = float(input("Enter impedance (2.0-8.0): "))
    
    # Categorical features
    phase_options = ['Single-phase', 'Three-phase']
    print("\nSelect phase:")
    for i, p in enumerate(phase_options):
        print(f"{i+1}. {p}")
    phase_choice = int(input("Enter choice (1-2): "))
    specs['phase'] = phase_options[phase_choice-1]
    
    frequency_options = [50, 60]
    print("\nSelect frequency (Hz):")
    for i, f in enumerate(frequency_options):
        print(f"{i+1}. {f}")
    frequency_choice = int(input("Enter choice (1-2): "))
    specs['frequency'] = frequency_options[frequency_choice-1]
    
    installation_options = ['Indoor', 'Outdoor']
    print("\nSelect installation type:")
    for i, inst in enumerate(installation_options):
        print(f"{i+1}. {inst}")
    installation_choice = int(input("Enter choice (1-2): "))
    specs['installation_type'] = installation_options[installation_choice-1]
    
    insulation_options = ['Oil', 'Dry']
    print("\nSelect insulation type:")
    for i, ins in enumerate(insulation_options):
        print(f"{i+1}. {ins}")
    insulation_choice = int(input("Enter choice (1-2): "))
    specs['insulation_type'] = insulation_options[insulation_choice-1]
    
    location_options = ['North America', 'Europe', 'Asia', 'South America']
    print("\nSelect manufacturing location:")
    for i, loc in enumerate(location_options):
        print(f"{i+1}. {loc}")
    location_choice = int(input("Enter choice (1-4): "))
    specs['manufacturing_location'] = location_options[location_choice-1]
    
    return specs

def predict_with_all_models(preprocessor, transformer_specs):
    """
    Predict transformer price using all available models
    """
    # First check if models directory exists locally or one level up
    if os.path.exists('models'):
        models_dir = 'models'
    else:
        models_dir = '../models'
        
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and 'preprocessor' not in f]
    
    predictions = {}
    
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '').replace('_', ' ').title()
        model_path = os.path.join(models_dir, model_file)
        
        try:
            model = load_model(model_path)
            predicted_price = predict_price(model, preprocessor, transformer_specs)
            predictions[model_name] = predicted_price
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    return predictions

def example_transformer_specs():
    """
    Create an example transformer specification
    """
    return {
        'power_rating': 1000,
        'primary_voltage': 13800,
        'secondary_voltage': 480,
        'core_weight': 1200,
        'copper_weight': 800,
        'insulation_weight': 200,
        'tank_weight': 950,
        'oil_weight': 1500,
        'total_weight': 4650,
        'cooling_type': 'ONAF',
        'tap_changer': True,
        'efficiency': 0.975,
        'impedance': 5.5,
        'phase': 'Three-phase',
        'frequency': 60,
        'installation_type': 'Outdoor',
        'insulation_type': 'Oil',
        'manufacturing_location': 'North America'
    }

def main():
    parser = argparse.ArgumentParser(description='Predict transformer price')
    parser.add_argument('--interactive', action='store_true', help='Get transformer specs interactively')
    parser.add_argument('--model', type=str, default='random_forest_regression', 
                        help='Model to use for prediction (default: random_forest_regression)')
    args = parser.parse_args()
    
    # Load preprocessor
    preprocessor = load_preprocessor()
    
    if args.interactive:
        # Get transformer specs from user
        transformer_specs = get_user_input()
    else:
        # Use example specs
        transformer_specs = example_transformer_specs()
        print("Using example transformer specifications:")
        for key, value in transformer_specs.items():
            print(f"  {key}: {value}")
    
    # Predict with specified model
    if os.path.exists(f"models/{args.model}.joblib"):
        model_path = f"models/{args.model}.joblib"
    else:
        model_path = f"../models/{args.model}.joblib"
        
    if os.path.exists(model_path):
        model = load_model(model_path)
        predicted_price = predict_price(model, preprocessor, transformer_specs)
        print(f"\nPredicted price using {args.model}: ${predicted_price:,.2f}")
    else:
        print(f"Model file not found: {model_path}")
    
    # Predict with all available models
    print("\nPredictions from all models:")
    all_predictions = predict_with_all_models(preprocessor, transformer_specs)
    
    # Sort by model name
    for model_name, price in sorted(all_predictions.items()):
        print(f"  {model_name}: ${price:,.2f}")
    
    # Calculate average prediction across all models
    if all_predictions:  # Check if the dictionary contains any values
        avg_price = np.mean(list(all_predictions.values()))
        print(f"\nAverage predicted price: ${avg_price:,.2f}")
    else:
        print("\nNo valid predictions available to calculate average.")

if __name__ == "__main__":
    main() 