import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic data for transformer price modeling
    
    Parameters:
    - n_samples: number of samples to generate
    
    Returns:
    - DataFrame with synthetic transformer data
    """
    np.random.seed(42)  # for reproducibility
    
    # Generate features with realistic ranges for transformers
    data = {
        # Power ratings (kVA)
        'power_rating': np.random.choice([25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000], n_samples),
        
        # Primary voltage (V)
        'primary_voltage': np.random.choice([4160, 12470, 13200, 13800, 24940, 34500], n_samples),
        
        # Secondary voltage (V)
        'secondary_voltage': np.random.choice([120, 208, 240, 277, 480, 600], n_samples),
        
        # Material weights (kg)
        'core_weight': np.random.uniform(100, 3000, n_samples),
        'copper_weight': np.random.uniform(50, 2000, n_samples),
        'insulation_weight': np.random.uniform(20, 500, n_samples),
        'tank_weight': np.random.uniform(100, 2500, n_samples),
        'oil_weight': np.random.uniform(200, 4000, n_samples),
        
        # Other specifications
        'cooling_type': np.random.choice(['ONAN', 'ONAF', 'OFAF', 'ODAF'], n_samples),
        'tap_changer': np.random.choice([True, False], n_samples),
        'efficiency': np.random.uniform(0.95, 0.99, n_samples),
        'impedance': np.random.uniform(2.0, 8.0, n_samples),
        
        # Categorical features
        'phase': np.random.choice(['Single-phase', 'Three-phase'], n_samples),
        'frequency': np.random.choice([50, 60], n_samples),
        'installation_type': np.random.choice(['Indoor', 'Outdoor'], n_samples),
        'insulation_type': np.random.choice(['Oil', 'Dry'], n_samples),
        
        # Manufacturing location factor (regional cost differences)
        'manufacturing_location': np.random.choice(['North America', 'Europe', 'Asia', 'South America'], n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate total weight
    df['total_weight'] = df['core_weight'] + df['copper_weight'] + df['insulation_weight'] + df['tank_weight'] + df['oil_weight']
    
    # Generate target variable (price) based on a complex combination of features with some noise
    # Base price is related to power rating
    base_price = df['power_rating'] * 100  
    
    # Material costs
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

if __name__ == "__main__":
    # Generate synthetic data
    transformer_data = generate_synthetic_data(1000)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV file
    transformer_data.to_csv('../data/transformer_data.csv', index=False)
    
    print(f"Generated {len(transformer_data)} transformer data samples and saved to '../data/transformer_data.csv'")
    print("Sample data:")
    print(transformer_data.head())
    print("\nData statistics:")
    print(transformer_data.describe()) 