"""
Manual Data Entry Template Generator

This script generates CSV templates for manually entering real transformer market data.
You can use these templates to collect data from Volza.com by manually copying information.
"""

from src.transformer_data_sample_generator import save_sample_data, print_data_entry_instructions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate templates for manual transformer data entry')
    parser.add_argument('--rows', type=int, default=50, help='Number of rows in the template')
    parser.add_argument('--sample', action='store_true', help='Include sample data as examples (default: empty template)')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    
    # Generate template or sample data
    if args.sample:
        print("Generating template with sample data examples...")
        save_sample_data(num_samples=args.rows, filename=args.output, template_only=False)
    else:
        print("Generating empty template for manual data entry...")
        save_sample_data(num_samples=args.rows, filename=args.output, template_only=True)
    
    # Print instructions
    print_data_entry_instructions()
    
    print("\nAdditional instructions:")
    print("1. Visit Volza.com and navigate to the transformer product pages")
    print("2. Look for tables with import/export data")
    print("3. Copy information manually into your CSV template")
    print("4. Once you have collected enough real data:")
    print("   - Run 'python src/market_data_integration.py' to process your data")
    print("   - Run 'python src/market_data_visualization.py' to generate visualizations") 