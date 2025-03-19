"""
Market Data Visualization

This module provides functions to visualize the scraped transformer market data,
including comparison with synthetic data to help understand the differences
between real market prices and model predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging

from src.market_data_integration import load_market_data, prepare_market_data, combine_with_synthetic_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def set_plot_style():
    """Set common plot style parameters"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 12


def plot_price_vs_power_rating(df, output_dir='visualizations'):
    """
    Plot transformer price vs power rating
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transformer data
    output_dir : str
        Directory to save visualizations
    """
    if df.empty or 'Power Rating (KVA)' not in df.columns or 'Unit Price (USD)' not in df.columns:
        logger.warning("Cannot create price vs power rating plot: missing required columns")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if we have data source information
    if 'Data Source' in df.columns:
        # Create scatter plot with different colors for different data sources
        sources = df['Data Source'].unique()
        for source in sources:
            subset = df[df['Data Source'] == source]
            ax.scatter(
                subset['Power Rating (KVA)'], 
                subset['Unit Price (USD)'],
                alpha=0.7,
                label=source
            )
    else:
        # Create simple scatter plot
        ax.scatter(df['Power Rating (KVA)'], df['Unit Price (USD)'], alpha=0.7)
    
    # Add trend line
    if len(df) > 1:
        # Use lowess smoother if available, otherwise use a simple polynomial fit
        try:
            sns.regplot(
                x='Power Rating (KVA)', 
                y='Unit Price (USD)', 
                data=df,
                scatter=False,
                lowess=True, 
                line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                ax=ax
            )
        except:
            # Fallback to polynomial fit
            z = np.polyfit(df['Power Rating (KVA)'], df['Unit Price (USD)'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(df['Power Rating (KVA)'].min(), df['Power Rating (KVA)'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', linewidth=2)
    
    # Customize plot
    ax.set_title('Transformer Price vs Power Rating', fontsize=16)
    ax.set_xlabel('Power Rating (KVA)', fontsize=14)
    ax.set_ylabel('Unit Price (USD)', fontsize=14)
    
    # Add log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if needed
    if 'Data Source' in df.columns:
        ax.legend(title='Data Source')
    
    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, f'price_vs_power_rating_{datetime.now().strftime("%Y%m%d")}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved price vs power rating plot to {filename}")
    plt.close()


def plot_price_per_kva(df, output_dir='visualizations'):
    """
    Plot price per KVA distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transformer data
    output_dir : str
        Directory to save visualizations
    """
    if df.empty or 'Power Rating (KVA)' not in df.columns or 'Unit Price (USD)' not in df.columns:
        logger.warning("Cannot create price per KVA plot: missing required columns")
        return
        
    # Calculate price per KVA
    df = df.copy()
    df['Price per KVA'] = df['Unit Price (USD)'] / df['Power Rating (KVA)']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if we have data source information
    if 'Data Source' in df.columns:
        sns.boxplot(
            x='Data Source', 
            y='Price per KVA',
            data=df,
            ax=ax
        )
        
        # Add individual points for better visibility
        sns.stripplot(
            x='Data Source', 
            y='Price per KVA',
            data=df,
            color='black',
            alpha=0.5,
            jitter=True,
            ax=ax
        )
    else:
        # Create simple boxplot
        sns.boxplot(y=df['Price per KVA'], ax=ax)
        sns.stripplot(y=df['Price per KVA'], color='black', alpha=0.5, jitter=True, ax=ax)
    
    # Customize plot
    ax.set_title('Price per KVA Distribution', fontsize=16)
    if 'Data Source' in df.columns:
        ax.set_xlabel('Data Source', fontsize=14)
    ax.set_ylabel('Price per KVA (USD)', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, f'price_per_kva_distribution_{datetime.now().strftime("%Y%m%d")}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved price per KVA distribution plot to {filename}")
    plt.close()


def plot_power_rating_distribution(df, output_dir='visualizations'):
    """
    Plot power rating distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transformer data
    output_dir : str
        Directory to save visualizations
    """
    if df.empty or 'Power Rating (KVA)' not in df.columns:
        logger.warning("Cannot create power rating distribution plot: missing required columns")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if we have data source information
    if 'Data Source' in df.columns:
        # Create histogram with different colors for different data sources
        sources = df['Data Source'].unique()
        
        # Create histogram with KDE
        for source in sources:
            subset = df[df['Data Source'] == source]
            sns.histplot(
                subset['Power Rating (KVA)'],
                kde=True,
                alpha=0.6,
                label=source,
                ax=ax
            )
    else:
        # Create simple histogram with KDE
        sns.histplot(df['Power Rating (KVA)'], kde=True, ax=ax)
    
    # Customize plot
    ax.set_title('Transformer Power Rating Distribution', fontsize=16)
    ax.set_xlabel('Power Rating (KVA)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add log scale for better visualization of wide range
    ax.set_xscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if needed
    if 'Data Source' in df.columns:
        ax.legend(title='Data Source')
    
    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, f'power_rating_distribution_{datetime.now().strftime("%Y%m%d")}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    logger.info(f"Saved power rating distribution plot to {filename}")
    plt.close()


def plot_country_analysis(df, output_dir='visualizations'):
    """
    Plot country analysis for origin and destination
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transformer data
    output_dir : str
        Directory to save visualizations
    """
    if df.empty:
        logger.warning("Cannot create country analysis plots: empty DataFrame")
        return
        
    # Origin countries plot
    if 'Origin' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Get top 10 origin countries
        top_origins = df['Origin'].value_counts().nlargest(10)
        
        # Create bar plot
        ax = sns.barplot(x=top_origins.index, y=top_origins.values)
        
        # Customize plot
        plt.title('Top 10 Origin Countries for Transformers', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Number of Transformers', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, count in enumerate(top_origins.values):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        # Save plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.join(output_dir, f'top_origin_countries_{datetime.now().strftime("%Y%m%d")}.png')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        logger.info(f"Saved origin countries plot to {filename}")
        plt.close()
    
    # Destination countries plot
    if 'Destination' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Get destination countries
        destinations = df['Destination'].value_counts()
        
        # Create bar plot
        ax = sns.barplot(x=destinations.index, y=destinations.values)
        
        # Customize plot
        plt.title('Destination Countries for Transformers', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Number of Transformers', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, count in enumerate(destinations.values):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        # Save plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.join(output_dir, f'destination_countries_{datetime.now().strftime("%Y%m%d")}.png')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        logger.info(f"Saved destination countries plot to {filename}")
        plt.close()
        
    # Price by origin country
    if 'Origin' in df.columns and 'Unit Price (USD)' in df.columns and 'Power Rating (KVA)' in df.columns:
        # Calculate price per KVA
        df = df.copy()
        df['Price per KVA'] = df['Unit Price (USD)'] / df['Power Rating (KVA)']
        
        # Get top 5 origins with enough data
        origin_counts = df['Origin'].value_counts()
        top_origins = origin_counts[origin_counts >= 3].nlargest(5).index.tolist()
        
        if top_origins:
            plt.figure(figsize=(14, 8))
            
            # Filter data
            filtered_df = df[df['Origin'].isin(top_origins)]
            
            # Create box plot
            ax = sns.boxplot(x='Origin', y='Price per KVA', data=filtered_df)
            
            # Add swarm plot for individual points
            sns.swarmplot(x='Origin', y='Price per KVA', data=filtered_df, color='black', alpha=0.5)
            
            # Customize plot
            plt.title('Price per KVA by Origin Country', fontsize=16)
            plt.xlabel('Origin Country', fontsize=14)
            plt.ylabel('Price per KVA (USD)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Save plot
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            filename = os.path.join(output_dir, f'price_by_origin_{datetime.now().strftime("%Y%m%d")}.png')
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            logger.info(f"Saved price by origin plot to {filename}")
            plt.close()


def create_visualizations(market_df=None, synthetic_df=None, output_dir='visualizations'):
    """
    Create all visualizations for market data and comparison with synthetic data
    
    Parameters:
    -----------
    market_df : pandas.DataFrame, optional
        Market data DataFrame. If None, will try to load from file.
    synthetic_df : pandas.DataFrame, optional
        Synthetic data DataFrame. If None, will try to load from file.
    output_dir : str
        Directory to save visualizations
    """
    # Set plot style
    set_plot_style()
    
    # Load market data if not provided
    if market_df is None:
        market_df = load_market_data()
        if not market_df.empty:
            market_df = prepare_market_data(market_df)
    
    # Load synthetic data if not provided and market data exists
    combined_df = None
    if not market_df.empty:
        if synthetic_df is None:
            # Try to load combined data
            try:
                combined_df = combine_with_synthetic_data(market_df)
            except Exception as e:
                logger.error(f"Error combining with synthetic data: {e}")
        else:
            # Use provided synthetic data
            synthetic_df['Data Source'] = 'Synthetic'
            
            # Ensure same columns in both datasets
            common_columns = set(market_df.columns).intersection(set(synthetic_df.columns))
            market_df = market_df[list(common_columns)]
            synthetic_df = synthetic_df[list(common_columns)]
            
            combined_df = pd.concat([market_df, synthetic_df], ignore_index=True)
    
    # Create visualizations for market data
    if not market_df.empty:
        logger.info("Creating visualizations for market data...")
        plot_power_rating_distribution(market_df, output_dir)
        plot_country_analysis(market_df, output_dir)
        
        if 'Unit Price (USD)' in market_df.columns and 'Power Rating (KVA)' in market_df.columns:
            plot_price_vs_power_rating(market_df, output_dir)
            plot_price_per_kva(market_df, output_dir)
    
    # Create comparison visualizations if combined data available
    if combined_df is not None and not combined_df.empty:
        logger.info("Creating comparison visualizations with synthetic data...")
        plot_power_rating_distribution(combined_df, output_dir)
        plot_price_vs_power_rating(combined_df, output_dir)
        plot_price_per_kva(combined_df, output_dir)


if __name__ == "__main__":
    # Example usage
    create_visualizations() 