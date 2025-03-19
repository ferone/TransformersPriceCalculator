# Transformer Price Calculator

A machine learning model to predict the price of electrical transformers based on specifications such as:
- Material weights (core, copper, insulation, etc.)
- Power rating
- Type of transformer
- Voltage ratings
- Other specifications

## Project Structure
- `data/`: Sample and real transformer data
- `models/`: Trained regression models
- `src/`: Source code
  - `data_processing.py`: Data cleaning and preparation
  - `feature_engineering.py`: Feature creation and selection
  - `model_training.py`: Train regression models
  - `model_evaluation.py`: Evaluate model performance
  - `predict.py`: Make predictions with trained models
  - `visualization.py`: Visualize data and results
  - `material_prices.py`: Real-time material price fetching
  - `transformer_data_scraper.py`: Scrape real transformer market data
  - `market_data_integration.py`: Integrate market data with synthetic data
  - `market_data_visualization.py`: Visualize market data comparison
- `notebooks/`: Jupyter notebooks for exploration
- `app.py`: Simple web application to use the model

## Setup
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the model training: `python src/model_training.py`
4. Make predictions: `python src/predict.py`

## Usage
The model requires inputs of transformer specifications to predict price. See example in `notebooks/example_usage.ipynb`. 

## How it works

### Transformer Price Calculator Overview

This is a machine learning-based calculator for electrical transformer prices. Let me provide a brief overview of what's been built:

#### A - Data Generation and Model Training
---------------------------------------

1. Synthetic Data Generation: The system creates realistic transformer data with parameters like power ratings, voltage levels, material weights, and other specifications.

2. Multiple Regression Models: The application trains several regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Random Forest
  - Gradient Boosting

3. Model Performance: According to the results, Gradient Boosting performed best with an R² of 0.9873 (98.73% of variance explained), followed by Random Forest with an R² of 0.9822.

#### B - Web Application Features
------------------------------
The Streamlit application provides:

1. Interactive Price Calculator: Users can input transformer specifications and get a price estimate.

2. Model Comparison: The "Model Performance" tab shows metrics for all models, helping users understand the prediction confidence.

3. Educational Information: The "About Transformers" tab explains electrical transformers and factors affecting their pricing.

#### C - How to Use the Application
-------------------------------

1. Access the app at http://localhost:8501 in your web browser.

2. Input transformer specifications such as:
  - Power rating
  - Voltage levels
  - Material weights (core, copper, insulation, etc.)
  - Other technical parameters

3. Select a model (Gradient Boosting is recommended based on performance)

4. Click "Calculate Transformer Price" to get a price estimate

### Material Cost Update Module
---------------------------

The project includes a comprehensive material cost module:

1. Material Prices Module:
  - Real-time material price fetching from the MetalpriceAPI (with a free tier option)
  - Fallback web scraping capability from London Metal Exchange for when API access isn't available
  - Local caching of material prices with staleness detection
  - Default material prices as a fallback when online sources aren't accessible
  - Conversion utilities to calculate material costs based on weights

2. Price Calculation Integration
  - The application shows current material prices in USD/Ton
  - When calculating a transformer price, it breaks down the raw material costs
  - It shows the percentage of the total price that comes from raw materials vs. labor/overhead/profit

3. UI Features
  - Price Calculator: Features the ML model selection, current material prices table, transformer specification form, and detailed price breakdown
  - Model Performance: Shows model comparison metrics and material price visualization
  - About Transformers: Provides educational information about transformers and material cost impacts

### Real Market Data Integration
---------------------------

The project now includes tools to gather and utilize real market data:

1. **Transformer Data Scraper**
   - Scrapes real transformer price data from Volza.com
   - Collects information from multiple countries and destinations
   - Extracts key details: country of origin, destination, date, HSN code, quantity, description, and price
   - Automatically extracts power ratings and other specifications from descriptions
   - Saves data in structured CSV format for model training

2. **Market Data Integration**
   - Extracts features from scraped descriptions (voltage, phase, frequency, etc.)
   - Combines real market data with synthetic data for model training
   - Preserves data source information for comparison
   - Provides analysis of price per KVA, origin countries, and more

3. **Market Data Visualization**
   - Creates visualizations comparing real market data with synthetic data
   - Shows price vs. power rating relationships
   - Displays price per KVA distributions
   - Analyzes country of origin and destination impact on prices
   - Generates insights on market trends and pricing factors

### Using the Scraper

To gather real transformer market data:

```bash
# Basic usage (uses default URLs)
python run_scraper.py

# Specify custom URLs to scrape
python run_scraper.py --urls https://www.volza.com/p/electrical-transformer/import/import-in-france/ https://www.volza.com/p/electrical-transformer/import/import-in-germany/

# Load existing scraped data without scraping again
python run_scraper.py --load-only
```

After scraping, you can integrate the data with your model:

```bash
# Run the market data integration to prepare data for model training
python src/market_data_integration.py

# Create visualizations comparing real and synthetic data
python src/market_data_visualization.py
```

## Possible Improvements
Some potential enhancements you might consider:

1. More Sources: Expand data scraping to additional sources for greater market coverage.
2. More Features: Add additional technical specifications that affect pricing.
3. Price History: Integrate a feature to track price changes over time.
4. Geographical Models: Train separate models for different regions to capture market differences.