"""
Transformer Data Cleaner

This script cleans the manually downloaded transformer data file,
identifying and extracting only valid transformer entries for use in model training.
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

def clean_price(price_str):
    """Clean and convert price string to float"""
    if pd.isna(price_str) or not isinstance(price_str, str):
        return None
    
    # Remove currency symbols, commas, and whitespace
    price_cleaned = re.sub(r'[$,\s]', '', price_str)
    
    try:
        return float(price_cleaned)
    except (ValueError, TypeError):
        return None

def extract_power_rating(description):
    """Extract power rating in KVA from transformer description"""
    if pd.isna(description) or not isinstance(description, str):
        return None
    
    description = description.upper()
    
    # Look for KVA
    kva_patterns = [
        r'(\d+(?:\.\d+)?)[\s-]*KVA',
        r'(\d+(?:\.\d+)?)[\s-]*K\.?V\.?A',
        r'(\d+(?:\.\d+)?)[\s-]*KILO\s*VOLT\s*AMPERE'
    ]
    
    for pattern in kva_patterns:
        match = re.search(pattern, description)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                continue
    
    # Look for MVA and convert to KVA
    mva_patterns = [
        r'(\d+(?:\.\d+)?)[\s-]*MVA',
        r'(\d+(?:\.\d+)?)[\s-]*M\.?V\.?A',
        r'(\d+(?:\.\d+)?)[\s-]*MEGA\s*VOLT\s*AMPERE'
    ]
    
    for pattern in mva_patterns:
        match = re.search(pattern, description)
        if match:
            try:
                # Convert MVA to KVA (1 MVA = 1000 KVA)
                return float(match.group(1)) * 1000
            except (ValueError, TypeError):
                continue
    
    return None

def extract_voltage(description):
    """Extract voltage information from transformer description"""
    if pd.isna(description) or not isinstance(description, str):
        return None, None
    
    description = description.upper()
    
    # Common voltage patterns like 11KV/433V or 33KV/11KV
    voltage_patterns = [
        r'(\d+(?:\.\d+)?)\s*KV\s*[/\\-]?\s*(\d+(?:\.\d+)?)\s*(KV|V)',
        r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*(KV|V)',
        r'(\d+(?:\.\d+)?)\s*KV\s*TO\s*(\d+(?:\.\d+)?)\s*(KV|V)'
    ]
    
    for pattern in voltage_patterns:
        match = re.search(pattern, description)
        if match:
            try:
                primary = float(match.group(1))  # Primary voltage
                secondary = float(match.group(2))  # Secondary value
                
                # Convert to kV if needed
                if match.group(3) == 'V':
                    secondary = secondary / 1000
                
                return primary, secondary
            except (ValueError, TypeError, IndexError):
                continue
    
    return None, None

def is_transformer_related(description):
    """Check if the description is related to electrical transformers"""
    if pd.isna(description) or not isinstance(description, str):
        return False
    
    description = description.lower()
    
    # Positive indicators (strong evidence of transformer)
    transformer_keywords = [
        'transformer', 'mvA ', 'kva ', 
        'power transformer', 'distribution transformer',
        'onan', 'onaf', 'oil immersed', 'dry type',
        '33kv', '11kv', '22kv', '132kv', '220kv', '400kv'
    ]
    
    # Exclude keywords for items that are not actually transformers
    exclude_keywords = [
        'beauty', 'microwave', 'hair dryer', 'led', 'power supply', 'powersupply',
        'charger', 'adapter', 'toy', 'heater', 'fan', 'motor', 'bulb', 'lamp',
        'toaster', 'blender', 'television', 'tv', 'remote', 'speaker', 'audio',
        'circuit board', 'pcb'
    ]
    
    # First check exclusions (stronger criteria)
    for keyword in exclude_keywords:
        if keyword in description:
            return False
    
    # Then check inclusions
    return any(keyword in description for keyword in transformer_keywords)

def clean_transformer_data(input_file, output_file=None):
    """
    Clean transformer data and extract valid entries
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the cleaned data. If None, a default name will be used.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing cleaned transformer data
    """
    print(f"Loading data from {input_file}...")
    
    # Try to read the file, handling potential encoding issues
    try:
        df = pd.read_csv(input_file)
    except UnicodeDecodeError:
        # Try with a different encoding if the default fails
        df = pd.read_csv(input_file, encoding='latin1')
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Remove empty rows (rows where all columns except maybe the first are NaN)
    df_cleaned = df_cleaned.dropna(thresh=2)
    
    print(f"Original data: {len(df)} rows")
    print(f"After removing empty rows: {len(df_cleaned)} rows")
    
    # Clean price column
    df_cleaned['Unit Price (USD)'] = df_cleaned['Unit Price (USD)'].apply(clean_price)
    
    # Extract power rating from description if not already filled
    missing_power_rating = df_cleaned['Power Rating (KVA)'].isna()
    df_cleaned.loc[missing_power_rating, 'Power Rating (KVA)'] = df_cleaned.loc[missing_power_rating, 'Description'].apply(extract_power_rating)
    
    # Extract voltage information
    primary_voltage, secondary_voltage = zip(*df_cleaned['Description'].apply(extract_voltage))
    df_cleaned['Primary Voltage (kV)'] = primary_voltage
    df_cleaned['Secondary Voltage (kV)'] = secondary_voltage
    
    # Add a column to indicate if the entry is transformer-related
    df_cleaned['Is Transformer'] = df_cleaned['Description'].apply(is_transformer_related)
    
    # Filter out non-transformer related entries
    df_transformer = df_cleaned[df_cleaned['Is Transformer']].copy()
    
    print(f"Identified transformer entries: {len(df_transformer)} rows")
    
    # Mark as real market data
    df_transformer['Data Source'] = 'Real (Market)'
    
    # If no output file is specified, create a default name
    if output_file is None:
        output_file = f"data/transformer_real_data_cleaned_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # Save the cleaned data
    df_transformer.to_csv(output_file, index=False)
    print(f"Saved cleaned transformer data to {output_file}")
    
    # Also create a combined file with both valid and invalid entries (for reference)
    reference_file = output_file.replace('.csv', '_reference.csv')
    df_cleaned.to_csv(reference_file, index=False)
    print(f"Saved reference file with all entries to {reference_file}")
    
    # Print summary statistics
    print("\n=== Transformer Data Summary ===")
    print(f"Total valid transformer entries: {len(df_transformer)}")
    
    if len(df_transformer) > 0:
        # Power rating statistics
        power_ratings = df_transformer['Power Rating (KVA)'].dropna()
        if len(power_ratings) > 0:
            print("\nPower Rating (KVA) statistics:")
            print(f"  Min: {power_ratings.min():.2f}")
            print(f"  Max: {power_ratings.max():.2f}")
            print(f"  Mean: {power_ratings.mean():.2f}")
            print(f"  Median: {power_ratings.median():.2f}")
        
        # Price statistics
        prices = df_transformer['Unit Price (USD)'].dropna()
        if len(prices) > 0:
            print("\nUnit Price (USD) statistics:")
            print(f"  Min: ${prices.min():.2f}")
            print(f"  Max: ${prices.max():.2f}")
            print(f"  Mean: ${prices.mean():.2f}")
            print(f"  Median: ${prices.median():.2f}")
        
        # Origin country distribution
        if 'Origin' in df_transformer.columns:
            origin_counts = df_transformer['Origin'].value_counts()
            print("\nTop origin countries:")
            for country, count in origin_counts.nlargest(5).items():
                print(f"  {country}: {count}")
    
    return df_transformer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean transformer data')
    parser.add_argument('--input', type=str, default='data/Transformers real data to clean.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Clean the data
    clean_transformer_data(args.input, args.output) 