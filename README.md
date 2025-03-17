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
- `notebooks/`: Jupyter notebooks for exploration
- `app.py`: Simple web application to use the model

## Setup
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the model training: `python src/model_training.py`
4. Make predictions: `python src/predict.py`

## Usage
The model requires inputs of transformer specifications to predict price. See example in `notebooks/example_usage.ipynb`. 

How it works:

Transformer Price Calculator Overview

This is a machine learning-based calculator for electrical transformer prices. Let me provide a brief overview of the built:

A - Data Generation and Model Training
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



B - Web Application Features
------------------------------
The Streamlit application provides:

1. Interactive Price Calculator: Users can input transformer specifications and get a price estimate.

2. Model Comparison: The "Model Performance" tab shows metrics for all models, helping users understand the prediction confidence.

3. Educational Information: The "About Transformers" tab explains electrical transformers and factors affecting their pricing.


C - How to Use the Application
-------------------------------

1. Access the app at http://localhost:8501 in your web browser.

2. Input transformer specifications such as:
  - Power rating
  - Voltage levels
  - Material weights (core, copper, insulation, etc.)
  - Other technical parameters

3. Select a model (Gradient Boosting is recommended based on performance)

4. Click "Calculate Transformer Price" to get a price estimate

Possible Improvements
Some potential enhancements you might consider:

1. Real Data: Replace synthetic data with real market data for greater accuracy.

2. More Features: Add additional technical specifications that affect pricing.
3. Price History: Integrate a feature to track price changes over time.
4. Material Price Integration: Link to current market prices of raw materials.
5. You can stop the Streamlit application when you're done by pressing Ctrl+C in the terminal or closing the terminal window