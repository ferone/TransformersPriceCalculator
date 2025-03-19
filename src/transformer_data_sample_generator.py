"""
Transformer Market Data Sample Generator

This module creates a sample CSV file template with the structure expected 
from real transformer market data. You can use this file to manually enter
real transformer data from the Volza website.

The file includes all the fields that would be extracted by the scraper:
- Date
- HSN Code
- Description
- Origin country
- Destination country
- Quantity
- Unit Price (USD)
- Power Rating (KVA)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

def generate_sample_data(num_samples=20):
    """
    Generate sample transformer market data
    
    Parameters:
    -----------
    num_samples : int
        Number of sample records to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample transformer data
    """
    # Random dates in the past 2 years
    today = datetime.now()
    dates = [(today - timedelta(days=random.randint(1, 730))).strftime('%d-%m-%Y') 
             for _ in range(num_samples)]
    
    # Common HSN codes for transformers
    hsn_codes = ['85042100', '85042210', '85042290', '85042300', '85043100', '85043200']
    
    # Sample descriptions of transformers
    descriptions = [
        "3 Phase 11KV/433V 500KVA Distribution Transformer Oil Filled",
        "Single Phase 33kV/11kV 1000KVA Power Transformer",
        "Three Phase 33kV/400V 2000kVA Oil Immersed Transformer",
        "Dry Type 11kV/433V 315KVA Cast Resin Transformer",
        "Distribution Transformer 11/0.433kV 100KVA Oil Filled",
        "Power Transformer 132/33kV 40MVA ONAN/ONAF",
        "22kV/415V 630KVA Three Phase Distribution Transformer",
        "Single Phase 19.1kV/240V 50KVA Pole Mounted Transformer",
        "Three Phase 33/6.6kV 5MVA Power Transformer ONAN Cooling",
        "11kV/433V 1250KVA Distribution Transformer ONAN"
    ]
    
    # Origin countries
    origin_countries = [
        "China", "India", "South Korea", "Germany", "Italy", 
        "United States", "Turkey", "Japan", "Brazil", "Spain"
    ]
    
    # Destination countries
    destination_countries = [
        "United Kingdom", "United States", "Germany", "France", 
        "Spain", "India", "Japan", "Canada", "Australia"
    ]
    
    # Generate sample data
    data = {
        'Date': random.choices(dates, k=num_samples),
        'HSN Code': random.choices(hsn_codes, k=num_samples),
        'Description': random.choices(descriptions, k=num_samples),
        'Origin': random.choices(origin_countries, k=num_samples),
        'Destination': random.choices(destination_countries, k=num_samples),
        'Quantity': np.random.randint(1, 50, size=num_samples),
        'Unit Price (USD)': [None] * num_samples,  # Empty for manual entry
        'Power Rating (KVA)': [None] * num_samples,  # Empty for manual entry
        'Source URL': ["https://www.volza.com/p/electrical-transformer/import/..." 
                      for _ in range(num_samples)]
    }
    
    # Extract power ratings from descriptions
    for i, desc in enumerate(data['Description']):
        if 'KVA' in desc or 'kVA' in desc or 'kva' in desc:
            # Extract power rating if present in description
            import re
            match = re.search(r'(\d+)(?:\.\d+)?(?:\s*)(?:KVA|kVA|kva)', desc)
            if match:
                data['Power Rating (KVA)'][i] = float(match.group(1))
        
        # For MVA descriptions, convert to KVA
        if 'MVA' in desc or 'mva' in desc:
            match = re.search(r'(\d+)(?:\.\d+)?(?:\s*)(?:MVA|mva)', desc)
            if match:
                data['Power Rating (KVA)'][i] = float(match.group(1)) * 1000
    
    # Add a flag for manual data entry
    data['Real Market Data'] = ['Yes/No (Update this)'] * num_samples
    
    return pd.DataFrame(data)

def generate_empty_template(num_rows=20):
    """
    Generate an empty template for manual data entry
    
    Parameters:
    -----------
    num_rows : int
        Number of empty rows to include
        
    Returns:
    --------
    pandas.DataFrame
        Empty DataFrame template
    """
    columns = [
        'Date', 'HSN Code', 'Description', 'Origin', 'Destination',
        'Quantity', 'Unit Price (USD)', 'Power Rating (KVA)', 'Real Market Data'
    ]
    
    # Create empty DataFrame
    df = pd.DataFrame(columns=columns)
    
    # Add empty rows
    for _ in range(num_rows):
        df = pd.concat([df, pd.DataFrame([["DD-MM-YYYY", "", "", "", "", "", "", "", "Yes/No"]], columns=columns)], 
                       ignore_index=True)
    
    return df

def save_sample_data(output_dir='data', filename=None, num_samples=20, template_only=False):
    """
    Save sample data to CSV file
    
    Parameters:
    -----------
    output_dir : str
        Directory to save data
    filename : str, optional
        Filename to save data to. If None, a default name will be used.
    num_samples : int
        Number of sample records to generate
    template_only : bool
        If True, generates only an empty template
        
    Returns:
    --------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate default filename if not provided
    if filename is None:
        if template_only:
            filename = f"transformer_market_data_template.csv"
        else:
            filename = f"transformer_market_data_sample_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # Generate data
    if template_only:
        df = generate_empty_template(num_samples)
    else:
        df = generate_sample_data(num_samples)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {'empty template' if template_only else 'sample data'} to {output_path}")
    return output_path

def print_data_entry_instructions():
    """Print instructions for manual data entry"""
    print("\n=== Manual Data Entry Instructions ===")
    print("1. Open the CSV file in Excel or another spreadsheet program")
    print("2. For each transformer you find on Volza.com:")
    print("   - Enter the date in DD-MM-YYYY format")
    print("   - Enter the HSN code (if available)")
    print("   - Copy the description exactly as shown")
    print("   - Enter origin and destination countries")
    print("   - Enter the quantity")
    print("   - Enter the unit price in USD")
    print("   - If the power rating isn't auto-extracted, enter it manually")
    print("   - Set 'Real Market Data' to 'Yes'")
    print("3. Save the file and run the market data integration script:")
    print("   python src/market_data_integration.py")
    print("\nExample website URLs to check:")
    print("- https://www.volza.com/p/electrical-transformer/import/import-in-united-kingdom/")
    print("- https://www.volza.com/p/electrical-transformer/import/import-in-united-states/")
    print("- https://www.volza.com/p/hsn-code-850423/import/import-in-spain/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample transformer market data')
    parser.add_argument('--rows', type=int, default=20, help='Number of rows to generate')
    parser.add_argument('--empty', action='store_true', help='Generate empty template only')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    
    # Generate and save data
    save_sample_data(num_samples=args.rows, filename=args.output, template_only=args.empty)
    
    # Print instructions
    print_data_entry_instructions() 