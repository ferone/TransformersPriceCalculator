"""
Market Data Integration

This module integrates real market data from the Volza scraper
with the Transformer Price Calculator model training pipeline.

It includes functions to:
1. Load and prepare scraped data
2. Extract features from transformer descriptions
3. Map market data to the format expected by the model
4. Combine real market data with synthetic data for training
"""

import pandas as pd
import numpy as np
import logging
import re
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(file_path=None):
    """
    Load scraped market data from file
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file with market data. If None, the latest file will be used.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing market data
    """
    if file_path is None:
        data_dir = 'data'
        files = [f for f in os.listdir(data_dir) if f.startswith('transformer_market_data_') and f.endswith('.csv')]
        
        if not files:
            logger.warning("No market data files found")
            return pd.DataFrame()
            
        # Sort by modification time (most recent first)
        latest_file = sorted(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)[0]
        file_path = os.path.join(data_dir, latest_file)
    
    logger.info(f"Loading market data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return pd.DataFrame()


def extract_voltage_from_description(description):
    """
    Extract voltage information from transformer description
    
    Parameters:
    -----------
    description : str
        Transformer description text
        
    Returns:
    --------
    tuple
        (primary_voltage, secondary_voltage) in kV, or (None, None) if not found
    """
    if not description or not isinstance(description, str):
        return None, None
        
    description = description.lower()
    
    # Common voltage patterns
    # Look for common formats like "11kV/433V" or "33 kV to 11 kV"
    voltage_pattern = r'(\d+\.?\d*)\s*kV\s*[/\\-]?\s*(\d+\.?\d*)\s*(kV|V)'
    match = re.search(voltage_pattern, description)
    
    if match:
        primary = float(match.group(1))  # Primary voltage in kV
        secondary = float(match.group(2))  # Secondary value
        
        # Convert to kV if needed
        if match.group(3).lower() == 'v':
            secondary = secondary / 1000
            
        return primary, secondary
    
    return None, None


def extract_phase_from_description(description):
    """
    Extract phase information from transformer description
    
    Parameters:
    -----------
    description : str
        Transformer description text
        
    Returns:
    --------
    str
        'Single-phase' or 'Three-phase' or None if not found
    """
    if not description or not isinstance(description, str):
        return None
        
    description = description.lower()
    
    if any(term in description for term in ['single phase', 'single-phase', '1-phase', '1 phase']):
        return 'Single-phase'
    elif any(term in description for term in ['three phase', 'three-phase', '3-phase', '3 phase']):
        return 'Three-phase'
        
    # Default to three-phase for power transformers (most common)
    if 'power transformer' in description or 'distribution transformer' in description:
        return 'Three-phase'
        
    return None


def extract_transformer_type(description):
    """
    Extract transformer type from description
    
    Parameters:
    -----------
    description : str
        Transformer description text
        
    Returns:
    --------
    str
        Transformer type or None if not found
    """
    if not description or not isinstance(description, str):
        return None
        
    description = description.lower()
    
    type_map = {
        'power transformer': 'Power',
        'distribution transformer': 'Distribution',
        'auto transformer': 'Auto',
        'instrument transformer': 'Instrument',
        'current transformer': 'Current',
        'voltage transformer': 'Voltage',
        'potential transformer': 'Potential',
        'dry type': 'Dry-Type',
        'oil-filled': 'Oil-Filled',
        'oil filled': 'Oil-Filled',
        'step-up': 'Step-Up',
        'step up': 'Step-Up',
        'step-down': 'Step-Down',
        'step down': 'Step-Down'
    }
    
    for key, value in type_map.items():
        if key in description:
            return value
            
    # Default to power transformer (most common)
    return 'Power'


def extract_frequency_from_description(description):
    """
    Extract frequency information from transformer description
    
    Parameters:
    -----------
    description : str
        Transformer description text
        
    Returns:
    --------
    float
        Frequency in Hz or None if not found
    """
    if not description or not isinstance(description, str):
        return None
        
    description = description.lower()
    
    frequency_pattern = r'(\d+\.?\d*)\s*(hz|hertz)'
    match = re.search(frequency_pattern, description)
    
    if match:
        return float(match.group(1))
    
    # Default to common frequencies based on regional hints
    if any(country in description.lower() for country in ['usa', 'canada', 'japan', 'united states', 'america']):
        return 60.0
    else:
        return 50.0  # Most of the world uses 50Hz


def prepare_market_data(market_df):
    """
    Prepare market data for model integration
    
    Parameters:
    -----------
    market_df : pandas.DataFrame
        Raw market data from scraper
        
    Returns:
    --------
    pandas.DataFrame
        Prepared market data with consistent features
    """
    if market_df.empty:
        return pd.DataFrame()
        
    logger.info("Preparing market data for model integration")
    
    # Create a copy to avoid modifying the original
    df = market_df.copy()
    
    # Filter to keep only rows with price and power rating
    df = df[df['Unit Price (USD)'].notna() & df['Power Rating (KVA)'].notna()]
    
    if df.empty:
        logger.warning("No valid data after filtering for price and power rating")
        return pd.DataFrame()
    
    # Extract additional features from description
    df['Primary Voltage (kV)'], df['Secondary Voltage (kV)'] = zip(
        *df['Description'].apply(extract_voltage_from_description)
    )
    
    df['Phase'] = df['Description'].apply(extract_phase_from_description)
    df['Type'] = df['Description'].apply(extract_transformer_type)
    df['Frequency (Hz)'] = df['Description'].apply(extract_frequency_from_description)
    
    # Add data source marker
    df['Data Source'] = 'Real (Market)'
    
    # Format for model integration
    model_columns = [
        'Power Rating (KVA)', 
        'Primary Voltage (kV)', 
        'Secondary Voltage (kV)', 
        'Phase', 
        'Type', 
        'Frequency (Hz)',
        'Unit Price (USD)',
        'Description',
        'Data Source'
    ]
    
    # Keep only columns useful for the model
    result_df = pd.DataFrame()
    for col in model_columns:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            # For missing columns, add with NaN values
            result_df[col] = np.nan
    
    # Fill missing values with defaults
    result_df['Frequency (Hz)'].fillna(50.0, inplace=True)
    result_df['Phase'].fillna('Three-phase', inplace=True)
    result_df['Type'].fillna('Power', inplace=True)
    
    logger.info(f"Prepared {len(result_df)} records for model integration")
    return result_df


def combine_with_synthetic_data(market_df, synthetic_data_path='data/transformer_synthetic_data.csv'):
    """
    Combine real market data with synthetic data for model training
    
    Parameters:
    -----------
    market_df : pandas.DataFrame
        Prepared market data
    synthetic_data_path : str
        Path to synthetic data file
        
    Returns:
    --------
    pandas.DataFrame
        Combined dataset for model training
    """
    if market_df.empty:
        logger.warning("No market data to combine with synthetic data")
        return pd.DataFrame()
        
    try:
        # Load synthetic data
        synth_df = pd.read_csv(synthetic_data_path)
        logger.info(f"Loaded {len(synth_df)} records of synthetic data")
        
        # Mark synthetic data
        synth_df['Data Source'] = 'Synthetic'
        
        # Ensure same columns in both datasets
        common_columns = set(market_df.columns).intersection(set(synth_df.columns))
        
        if not common_columns:
            logger.error("No common columns between market and synthetic data")
            return market_df
            
        # Combine datasets
        combined_df = pd.concat([market_df[common_columns], synth_df[common_columns]], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} records")
        
        # Save combined dataset
        output_path = f"data/transformer_combined_data_{datetime.now().strftime('%Y%m%d')}.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved combined dataset to {output_path}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error combining data: {e}")
        return market_df
        

def analyze_market_data(market_df):
    """
    Analyze market data to extract insights
    
    Parameters:
    -----------
    market_df : pandas.DataFrame
        Market data DataFrame
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    if market_df.empty:
        return {}
        
    results = {}
    
    # Price per KVA analysis
    if 'Power Rating (KVA)' in market_df.columns and 'Unit Price (USD)' in market_df.columns:
        market_df['Price per KVA'] = market_df['Unit Price (USD)'] / market_df['Power Rating (KVA)']
        
        price_per_kva = market_df['Price per KVA'].dropna()
        
        results['price_per_kva'] = {
            'mean': price_per_kva.mean(),
            'median': price_per_kva.median(),
            'min': price_per_kva.min(),
            'max': price_per_kva.max(),
            'std': price_per_kva.std()
        }
        
    # Origin country analysis
    if 'Origin' in market_df.columns:
        origin_counts = market_df['Origin'].value_counts()
        results['origin_countries'] = origin_counts.to_dict()
        
    # Destination country analysis
    if 'Destination' in market_df.columns:
        destination_counts = market_df['Destination'].value_counts()
        results['destination_countries'] = destination_counts.to_dict()
        
    # Power rating distribution
    if 'Power Rating (KVA)' in market_df.columns:
        power_ratings = market_df['Power Rating (KVA)'].dropna()
        
        results['power_rating'] = {
            'mean': power_ratings.mean(),
            'median': power_ratings.median(),
            'min': power_ratings.min(),
            'max': power_ratings.max(),
            'std': power_ratings.std()
        }
        
    return results


if __name__ == "__main__":
    # Example usage
    market_df = load_market_data()
    
    if not market_df.empty:
        prepared_df = prepare_market_data(market_df)
        combined_df = combine_with_synthetic_data(prepared_df)
        analysis = analyze_market_data(prepared_df)
        
        # Print analysis results
        if analysis:
            print("\n==== Market Data Analysis ====")
            
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
                for country, count in sorted(analysis['origin_countries'].items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {country}: {count}")
                    
            if 'destination_countries' in analysis:
                print("\nDestination Countries:")
                for country, count in analysis['destination_countries'].items():
                    print(f"  {country}: {count}")
    else:
        print("No market data available. Run the scraper first.") 