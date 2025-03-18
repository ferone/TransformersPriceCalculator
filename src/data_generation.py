import pandas as pd
import numpy as np
import os
from weight_estimator import estimate_weights_from_power_and_voltage

def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic data for transformer price modeling with realistic weight scaling
    based on power ratings and voltage levels.
    
    Parameters:
    - n_samples: number of samples to generate
    
    Returns:
    - DataFrame with synthetic transformer data
    """
    np.random.seed(42)  # for reproducibility
    
    # Generate power ratings and voltage levels first
    power_ratings = np.random.choice([25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000], n_samples)
    primary_voltages = np.random.choice([4.16, 12.47, 13.2, 13.8, 24.94, 34.5], n_samples)  # in kV
    secondary_voltages = np.random.choice([0.12, 0.208, 0.24, 0.277, 0.48, 0.6], n_samples)  # in kV
    phases = np.random.choice(['Single-phase', 'Three-phase'], n_samples)
    
    # Initialize weight columns
    core_weights = np.zeros(n_samples)
    copper_weights = np.zeros(n_samples)
    insulation_weights = np.zeros(n_samples)
    tank_weights = np.zeros(n_samples)
    oil_weights = np.zeros(n_samples)
    total_weights = np.zeros(n_samples)
    
    # Calculate realistic weights based on power ratings and voltage levels
    for i in range(n_samples):
        weights = estimate_weights_from_power_and_voltage(
            power_ratings[i], 
            primary_voltages[i], 
            secondary_voltages[i],
            phases[i]
        )
        
        # Add some random variation (±10%) to make the data more realistic
        variation = np.random.uniform(0.9, 1.1, 6)
        
        core_weights[i] = weights['core_weight'] * variation[0]
        copper_weights[i] = weights['copper_weight'] * variation[1]
        insulation_weights[i] = weights['insulation_weight'] * variation[2]
        tank_weights[i] = weights['tank_weight'] * variation[3]
        oil_weights[i] = weights['oil_weight'] * variation[4]
        total_weights[i] = weights['total_weight'] * variation[5]
    
    # Create base data dictionary
    data = {
        # Power ratings (kVA)
        'power_rating': power_ratings,
        
        # Voltage levels (V)
        'primary_voltage': primary_voltages * 1000,  # Convert kV to V
        'secondary_voltage': secondary_voltages * 1000,  # Convert kV to V
        
        # Material weights (kg) - now based on realistic scaling
        'core_weight': core_weights,
        'copper_weight': copper_weights,
        'insulation_weight': insulation_weights,
        'tank_weight': tank_weights,
        'oil_weight': oil_weights,
        'total_weight': total_weights,
        
        # Other specifications
        'cooling_type': np.random.choice(['ONAN', 'ONAF', 'OFAF', 'ODAF'], n_samples),
        'tap_changer': np.random.choice([True, False], n_samples),
        'efficiency': np.random.uniform(0.95, 0.99, n_samples),
        'impedance': np.random.uniform(2.0, 8.0, n_samples),
        
        # Categorical features
        'phase': phases,
        'frequency': np.random.choice([50, 60], n_samples),
        'installation_type': np.random.choice(['Indoor', 'Outdoor'], n_samples),
        'insulation_type': np.random.choice(['Oil', 'Dry'], n_samples),
        
        # Manufacturing location factor (regional cost differences)
        'manufacturing_location': np.random.choice(['North America', 'Europe', 'Asia', 'South America'], n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (price) based on a complex combination of features with some noise
    # Base price is related to power rating
    base_price = df['power_rating'] * 100  
    
    # Material costs - use actual weights instead of random values
    material_cost = (df['core_weight'] * 5 +  # Core material cost
                     df['copper_weight'] * 15 +  # Copper is expensive
                     df['insulation_weight'] * 8 +
                     df['tank_weight'] * 3 +
                     df['oil_weight'] * 2)
    
    # Adjustments based on other factors
    adjustments = (
        df['efficiency'] * 20000 +  # Higher efficiency costs more
        (df['cooling_type'] == 'ONAF').astype(int) * 5000 +  # ONAF cooling adds cost
        (df['cooling_type'] == 'OFAF').astype(int) * 10000 +  # OFAF cooling adds more cost
        (df['cooling_type'] == 'ODAF').astype(int) * 15000 +  # ODAF cooling is most expensive
        df['tap_changer'].astype(int) * 7500 +  # Tap changer adds cost
        (df['phase'] == 'Three-phase').astype(int) * 5000 +  # Three-phase costs more
        (df['insulation_type'] == 'Dry').astype(int) * 8000  # Dry type costs more
    )
    
    # Manufacturing location adjustments
    location_factors = {
        'North America': 1.2,
        'Europe': 1.3,
        'Asia': 0.85,
        'South America': 0.95
    }
    location_multiplier = df['manufacturing_location'].map(location_factors)
    
    # Final price calculation with some random noise
    df['price'] = (base_price + material_cost + adjustments) * location_multiplier * (1 + np.random.normal(0, 0.05, n_samples))
    
    # Round price to nearest dollar
    df['price'] = np.round(df['price'], 0)
    
    return df

def verify_weight_scaling(data):
    """
    Verify the relationship between power ratings and weights
    
    Parameters:
    - data: DataFrame with transformer data
    
    Returns:
    - DataFrame with average weights by power rating
    """
    # Group by power rating and calculate average weights
    weight_by_power = data.groupby('power_rating').agg({
        'core_weight': 'mean',
        'copper_weight': 'mean', 
        'insulation_weight': 'mean',
        'tank_weight': 'mean',
        'oil_weight': 'mean',
        'total_weight': 'mean'
    }).reset_index()
    
    print("Weight scaling verification:")
    print(weight_by_power.sort_values('power_rating'))
    
    return weight_by_power

if __name__ == "__main__":
    # Generate synthetic data
    transformer_data = generate_synthetic_data(1000)
    
    # Verify weight scaling
    weight_scaling = verify_weight_scaling(transformer_data)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV file
    transformer_data.to_csv('../data/transformer_data.csv', index=False)
    
    # Save weight scaling data
    weight_scaling.to_csv('../data/weight_scaling.csv', index=False)
    
    print(f"Generated {len(transformer_data)} transformer data samples and saved to '../data/transformer_data.csv'")
    print("Sample data:")
    print(transformer_data.head())
    print("\nData statistics:")
    print(transformer_data.describe()) 