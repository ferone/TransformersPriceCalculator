"""
Integrate Real Transformer Data

This script integrates the cleaned real transformer market data with synthetic data
for use in the transformer price calculator model.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from market data integration module
from src.market_data_integration import prepare_market_data, combine_with_synthetic_data, analyze_market_data
from src.market_data_visualization import plot_price_vs_power_rating, plot_price_per_kva, plot_power_rating_distribution

# Import the quantity cleaner
from clean_transformer_quantity import clean_transformer_quantity_price

def integrate_real_transformer_data(real_data_file, synthetic_data_file=None, output_dir='data', min_power_rating=200):
    """
    Integrate real transformer data with synthetic data
    
    Parameters:
    -----------
    real_data_file : str
        Path to the cleaned real transformer data file
    synthetic_data_file : str, optional
        Path to the synthetic data file. If None, a default synthetic data file will be used.
    output_dir : str
        Directory to save the integrated data
    min_power_rating : float
        Minimum power rating (KVA) to keep in the dataset
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the integrated data
    """
    print(f"Loading and cleaning real transformer data from {real_data_file}...")
    
    # First clean the quantity and price data, filtering entries without power ratings or with low power ratings
    timestamp = datetime.now().strftime('%Y%m%d')
    quantity_cleaned_file = os.path.join(output_dir, f"transformer_real_data_cleaned_{timestamp}.csv")
    real_df = clean_transformer_quantity_price(real_data_file, quantity_cleaned_file, min_power_rating)
    
    print(f"Using cleaned data from {quantity_cleaned_file} for integration")
    
    # Default synthetic data file
    if synthetic_data_file is None:
        synthetic_data_file = 'data/transformer_synthetic_data.csv'
        # If the default file doesn't exist, create sample synthetic data
        if not os.path.exists(synthetic_data_file):
            print(f"Creating sample synthetic data as {synthetic_data_file} doesn't exist...")
            synthetic_df = create_sample_synthetic_data(min_power_rating=min_power_rating)
            synthetic_df.to_csv(synthetic_data_file, index=False)
            print(f"Saved sample synthetic data to {synthetic_data_file}")
        else:
            print(f"Loading synthetic data from {synthetic_data_file}...")
            synthetic_df = pd.read_csv(synthetic_data_file)
            # Filter synthetic data as well to keep consistent with real data
            synthetic_df = synthetic_df[synthetic_df['Power Rating (KVA)'] >= min_power_rating]
    else:
        print(f"Loading synthetic data from {synthetic_data_file}...")
        synthetic_df = pd.read_csv(synthetic_data_file)
        # Filter synthetic data as well to keep consistent with real data
        synthetic_df = synthetic_df[synthetic_df['Power Rating (KVA)'] >= min_power_rating]
    
    # Prepare real data
    print("Preparing real transformer data...")
    prepared_real_df = prepare_real_data(real_df)
    
    # Combine with synthetic data
    print("Combining real and synthetic data...")
    synthetic_df['Data Source'] = 'Synthetic'
    
    # Ensure same columns in both datasets
    common_columns = list(set(prepared_real_df.columns) & set(synthetic_df.columns))
    prepared_real_df = prepared_real_df[common_columns]
    synthetic_df = synthetic_df[common_columns]
    
    # Check if both dataframes have rows
    if prepared_real_df.empty:
        print("Warning: No real data available for integration")
        integrated_df = synthetic_df
    elif synthetic_df.empty:
        print("Warning: No synthetic data available for integration")
        integrated_df = prepared_real_df
    else:
        integrated_df = pd.concat([prepared_real_df, synthetic_df], ignore_index=True)
    
    # Save integrated data
    output_path = os.path.join(output_dir, f"transformer_integrated_data_{timestamp}.csv")
    integrated_df.to_csv(output_path, index=False)
    print(f"Saved integrated data to {output_path}")
    
    # Analyze data
    print("\n=== Real Market Data Analysis ===")
    real_analysis = analyze_market_data(prepared_real_df)
    print_analysis_results(real_analysis)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualization_dir = 'visualizations'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    # Create comparison visualizations
    plot_price_vs_power_rating(integrated_df, visualization_dir)
    plot_price_per_kva(integrated_df, visualization_dir)
    plot_power_rating_distribution(integrated_df, visualization_dir)
    
    # Create separate visualizations for real data only
    real_vis_dir = os.path.join(visualization_dir, 'real_data')
    if not os.path.exists(real_vis_dir):
        os.makedirs(real_vis_dir)
    
    plot_price_vs_power_rating(prepared_real_df, real_vis_dir)
    plot_price_per_kva(prepared_real_df, real_vis_dir)
    plot_power_rating_distribution(prepared_real_df, real_vis_dir)
    
    return integrated_df

def prepare_real_data(real_df):
    """
    Prepare real transformer data for integration
    
    Parameters:
    -----------
    real_df : pandas.DataFrame
        DataFrame containing real transformer data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing prepared real transformer data
    """
    # Create a copy to avoid modifying the original
    df = real_df.copy()
    
    # Filter to keep only rows with price and power rating
    df = df[df['Unit Price (USD)'].notna() & df['Power Rating (KVA)'].notna()]
    
    if df.empty:
        print("Warning: No valid data after filtering for price and power rating")
        return pd.DataFrame()
    
    # Make sure the data source is clearly marked
    df['Data Source'] = 'Real (Market)'
    
    # Fill missing voltage values with defaults
    if 'Primary Voltage (kV)' in df.columns and 'Secondary Voltage (kV)' in df.columns:
        # Common default values
        df['Primary Voltage (kV)'].fillna(11.0, inplace=True)
        df['Secondary Voltage (kV)'].fillna(0.433, inplace=True)
    
    # Fill missing phase information
    if 'Phase' not in df.columns:
        df['Phase'] = 'Three-phase'  # Default to three-phase (most common)
    
    # Fill missing transformer type
    if 'Type' not in df.columns:
        df['Type'] = 'Power'  # Default to power transformer
    
    # Fill missing frequency information
    if 'Frequency (Hz)' not in df.columns:
        df['Frequency (Hz)'] = 50.0  # Default to 50Hz (most common worldwide)
    
    print(f"Prepared {len(df)} records of real transformer data")
    return df

def create_sample_synthetic_data(num_samples=100, min_power_rating=200):
    """
    Create sample synthetic transformer data
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    min_power_rating : float
        Minimum power rating (KVA) to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic transformer data
    """
    # Power ratings (KVA) - adjusted to respect minimum power rating
    power_ratings = np.concatenate([
        np.random.uniform(min_power_rating, 1000, size=int(num_samples * 0.3)),
        np.random.uniform(1000, 10000, size=int(num_samples * 0.4)),
        np.random.uniform(10000, 100000, size=int(num_samples * 0.3))
    ])
    
    # Primary voltages (kV)
    primary_voltages = np.random.choice([6.6, 11, 22, 33, 66, 132, 220, 400], size=num_samples)
    
    # Secondary voltages (kV)
    secondary_voltages = np.random.choice([0.22, 0.4, 0.433, 0.69, 3.3, 6.6, 11, 33], size=num_samples)
    
    # Phases
    phases = np.random.choice(['Single-phase', 'Three-phase'], size=num_samples, p=[0.1, 0.9])
    
    # Types
    types = np.random.choice(['Power', 'Distribution', 'Auto', 'Instrument'], size=num_samples, p=[0.5, 0.4, 0.05, 0.05])
    
    # Frequencies
    frequencies = np.random.choice([50.0, 60.0], size=num_samples, p=[0.8, 0.2])
    
    # Unit prices (USD)
    # Approximate price formula: $20 * power rating + random variation
    unit_prices = 20 * power_ratings * (1 + np.random.uniform(-0.2, 0.3, size=num_samples))
    
    # Create synthetic DataFrame
    synthetic_data = {
        'Power Rating (KVA)': power_ratings,
        'Primary Voltage (kV)': primary_voltages,
        'Secondary Voltage (kV)': secondary_voltages,
        'Phase': phases,
        'Type': types,
        'Frequency (Hz)': frequencies,
        'Unit Price (USD)': unit_prices,
        'Data Source': ['Synthetic'] * num_samples
    }
    
    return pd.DataFrame(synthetic_data)

def print_analysis_results(analysis):
    """Print analysis results in a readable format"""
    if not analysis:
        print("No analysis results available")
        return
    
    if 'price_per_kva' in analysis:
        print("\nPrice per KVA (USD):")
        for key, value in analysis['price_per_kva'].items():
            print(f"  {key.capitalize()}: ${value:.2f}")
    
    if 'power_rating' in analysis:
        print("\nPower Rating (KVA):")
        for key, value in analysis['power_rating'].items():
            print(f"  {key.capitalize()}: {value:.2f}")
    
    if 'origin_countries' in analysis:
        print("\nTop 5 Origin Countries:")
        for country, count in sorted(list(analysis['origin_countries'].items())[:5], 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {country}: {count}")
    
    if 'destination_countries' in analysis:
        print("\nDestination Countries:")
        for country, count in analysis['destination_countries'].items():
            print(f"  {country}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate real transformer data with synthetic data')
    parser.add_argument('--real', type=str, default='data/transformer_real_data_cleaned_20250319.csv',
                        help='Path to the cleaned real transformer data file')
    parser.add_argument('--synthetic', type=str, default=None,
                        help='Path to the synthetic data file (optional)')
    parser.add_argument('--min-power', type=float, default=200,
                        help='Minimum power rating (KVA) to keep in the dataset')
    
    args = parser.parse_args()
    
    # Integrate data
    integrate_real_transformer_data(args.real, args.synthetic, min_power_rating=args.min_power) 