"""
Transformer Market Data Scraper Runner

This script runs the transformer data scraper to collect real market data
from Volza.com for use in training and validating the Transformer Price Calculator model.
"""

import os
import pandas as pd
import argparse
from src.transformer_data_scraper import scrape_transformer_data, load_scraped_data


def main():
    parser = argparse.ArgumentParser(description='Scrape transformer market data from Volza')
    parser.add_argument('--urls', nargs='+', help='URLs to scrape (optional)')
    parser.add_argument('--load-only', action='store_true', help='Only load existing data without scraping')
    parser.add_argument('--output', type=str, default=None, help='Output filename (optional)')
    args = parser.parse_args()

    if args.load_only:
        print("Loading existing scraped data...")
        df = load_scraped_data()
        if df.empty:
            print("No existing data found. Run without --load-only to scrape new data.")
            return
    else:
        print("Scraping transformer market data from Volza...")
        df = scrape_transformer_data(args.urls)
        
        if df.empty:
            print("No transformer data was scraped")
            return
            
    # Display summary statistics
    print(f"\n==== Transformer Market Data Summary ====")
    print(f"Total records: {len(df)}")
    
    # Count by destination country
    print("\nRecords by destination country:")
    country_counts = df['Destination'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country}: {count}")
    
    # Summarize power ratings
    if 'Power Rating (KVA)' in df.columns and df['Power Rating (KVA)'].notna().any():
        power_ratings = df['Power Rating (KVA)'].dropna()
        print("\nPower Rating (KVA) statistics:")
        print(f"  Min: {power_ratings.min():.2f}")
        print(f"  Max: {power_ratings.max():.2f}")
        print(f"  Mean: {power_ratings.mean():.2f}")
        print(f"  Median: {power_ratings.median():.2f}")
        
    # Summarize prices
    if 'Unit Price (USD)' in df.columns and df['Unit Price (USD)'].notna().any():
        prices = df['Unit Price (USD)'].dropna()
        print("\nUnit Price (USD) statistics:")
        print(f"  Min: ${prices.min():.2f}")
        print(f"  Max: ${prices.max():.2f}")
        print(f"  Mean: ${prices.mean():.2f}")
        print(f"  Median: ${prices.median():.2f}")
    
    # Show sample data
    print("\nSample data (first 5 rows):")
    sample_columns = ['Date', 'Description', 'Origin', 'Destination', 
                     'Quantity', 'Unit Price (USD)', 'Power Rating (KVA)']
    sample_df = df[sample_columns].head()
    
    # Format display for better readability
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 120)
    print(sample_df)
    
    # Reset display options
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.width')


if __name__ == "__main__":
    main() 