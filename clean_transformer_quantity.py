"""
Transformer Quantity and Price Cleaner

This script cleans the quantity field in transformer data, converting all quantity values 
to integers and recalculating unit prices when quantities > 1.
It also removes entries without power ratings and those with power ratings below 200 KVA.
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

def extract_quantity(quantity_str):
    """
    Extract numeric quantity from string with various formats
    Examples: "1", "2 NOS", "284 UNT", "NaN PAC", etc.
    """
    if pd.isna(quantity_str):
        return 1  # Default to 1 if no quantity specified
    
    if isinstance(quantity_str, (int, float)):
        if np.isnan(quantity_str):
            return 1
        return int(quantity_str)
    
    # Extract the number from the string using regex
    match = re.search(r'^(\d+)', str(quantity_str).strip())
    if match:
        return int(match.group(1))
    
    return 1  # Default to 1 if no numeric value found

def clean_transformer_quantity_price(input_file, output_file=None, min_power_rating=200):
    """
    Clean transformer data quantity field and recalculate unit prices
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the cleaned data. If None, a default name will be used.
    min_power_rating : float, optional
        Minimum power rating (KVA) to keep in the dataset
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing cleaned transformer data with proper quantities and unit prices
    """
    print(f"Loading data from {input_file}...")
    
    # Try to read the file, handling potential encoding issues
    try:
        df = pd.read_csv(input_file)
    except UnicodeDecodeError:
        # Try with a different encoding if the default fails
        df = pd.read_csv(input_file, encoding='latin1')
    
    print(f"Original data: {len(df)} rows")
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Add new column for cleaned quantity
    df_cleaned['Quantity (Integer)'] = df_cleaned['Quantity'].apply(extract_quantity)
    
    # Create original unit price column (preserve original prices)
    df_cleaned['Original Unit Price (USD)'] = df_cleaned['Unit Price (USD)']
    
    # Recalculate unit price when quantity > 1
    mask = (df_cleaned['Quantity (Integer)'] > 1) & df_cleaned['Unit Price (USD)'].notna()
    df_cleaned.loc[mask, 'Unit Price (USD)'] = df_cleaned.loc[mask, 'Unit Price (USD)'] / df_cleaned.loc[mask, 'Quantity (Integer)']
    
    # Filter out entries without power ratings
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Power Rating (KVA)'].notna()]
    missing_power_count = initial_count - len(df_cleaned)
    
    # Filter out entries with power ratings lower than specified minimum
    power_before_filter = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Power Rating (KVA)'] >= min_power_rating]
    low_power_count = power_before_filter - len(df_cleaned)
    
    # Drop the original Quantity column and rename the new one
    df_cleaned = df_cleaned.drop(columns=['Quantity'])
    df_cleaned = df_cleaned.rename(columns={'Quantity (Integer)': 'Quantity'})
    
    # If no output file is specified, create a default name
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_file = f"data/transformer_real_data_cleaned_{timestamp}.csv"
    
    # Save the cleaned data
    df_cleaned.to_csv(output_file, index=False)
    print(f"Saved cleaned transformer data to {output_file}")
    
    # Print summary statistics
    print("\n=== Transformer Data Cleaning Summary ===")
    print(f"Total entries: {len(df_cleaned)}")
    print(f"Entries with quantity > 1: {len(df_cleaned[df_cleaned['Quantity'] > 1])}")
    print(f"Entries removed due to missing power rating: {missing_power_count}")
    print(f"Entries removed due to power rating < {min_power_rating} KVA: {low_power_count}")
    
    # Price comparison statistics (before and after recalculation)
    price_diff = df_cleaned[df_cleaned['Quantity'] > 1]['Original Unit Price (USD)'] - df_cleaned[df_cleaned['Quantity'] > 1]['Unit Price (USD)']
    if len(price_diff) > 0:
        print("\nPrice adjustment statistics for entries with quantity > 1:")
        print(f"  Number of prices adjusted: {len(price_diff)}")
        print(f"  Average price reduction: ${price_diff.mean():.2f}")
        print(f"  Maximum price reduction: ${price_diff.max():.2f}")
    
    return df_cleaned

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean transformer data quantity field and recalculate unit prices')
    parser.add_argument('--input', type=str, default='data/transformer_real_data_cleaned_20250319.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--min-power', type=float, default=200,
                        help='Minimum power rating (KVA) to keep in the dataset')
    
    args = parser.parse_args()
    
    # Clean the data
    clean_transformer_quantity_price(args.input, args.output, args.min_power) 