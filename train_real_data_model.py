"""
Train Transformer Price Prediction Model Using Real Market Data

This script trains machine learning models to predict transformer prices
using the cleaned real market data.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Set Seaborn style
sns.set(style="whitegrid")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    mask = y_true != 0  # Avoid division by zero
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def load_real_data(data_path):
    """
    Load the cleaned real transformer data
    """
    print(f"Loading data from {data_path}")
    return pd.read_csv(data_path)

def preprocess_data(df, target_column='Unit Price (USD)', test_size=0.2, random_state=42):
    """
    Preprocess transformer data for machine learning
    
    Parameters:
    - df: DataFrame containing transformer data
    - target_column: The name of the price column
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed training and testing data
    - preprocessor: Fitted preprocessor pipeline
    """
    # Copy to avoid modifying the original
    df_copy = df.copy()
    
    # Print initial column information
    print("\nColumns in the dataset:")
    for col in df_copy.columns:
        print(f"  {col}: {df_copy[col].dtype} (Missing: {df_copy[col].isna().sum()})")
    
    # Rename the target column to 'price' for consistency
    df_copy = df_copy.rename(columns={target_column: 'price'})
    
    # Drop rows with missing price values
    initial_rows = len(df_copy)
    df_copy = df_copy.dropna(subset=['price'])
    dropped_price_rows = initial_rows - len(df_copy)
    print(f"\nDropped {dropped_price_rows} rows with missing price values. Remaining rows: {len(df_copy)}")
    
    # Check for zero or negative prices
    zero_or_negative_prices = (df_copy['price'] <= 0).sum()
    if zero_or_negative_prices > 0:
        print(f"Found {zero_or_negative_prices} rows with zero or negative prices. Removing these rows.")
        df_copy = df_copy[df_copy['price'] > 0]
    
    # Select relevant features only
    selected_features = [
        'Power Rating (KVA)', 
        'Primary Voltage (kV)', 
        'Secondary Voltage (kV)', 
        'Phase', 
        'Type', 
        'Frequency (Hz)',
        'price'
    ]
    
    # Keep only selected features that exist in the dataset
    available_features = [col for col in selected_features if col in df_copy.columns]
    print(f"\nSelected features that are available: {available_features}")
    df_filtered = df_copy[available_features].copy()
    
    # Print missing value information
    print("\nMissing values in filtered dataset:")
    for col in df_filtered.columns:
        print(f"  {col}: {df_filtered[col].isna().sum()} missing values")
    
    # Fill missing values in 'Frequency (Hz)' with 50.0 (most common worldwide frequency)
    if 'Frequency (Hz)' in df_filtered.columns:
        df_filtered['Frequency (Hz)'].fillna(50.0, inplace=True)
    
    # Ensure categorical columns exist and fill missing with defaults
    if 'Phase' not in df_filtered.columns:
        df_filtered['Phase'] = 'Three-phase'  # Default to three-phase (most common)
    elif df_filtered['Phase'].isna().any():
        df_filtered['Phase'].fillna('Three-phase', inplace=True)
    
    if 'Type' not in df_filtered.columns:
        df_filtered['Type'] = 'Power'  # Default to power transformer
    elif df_filtered['Type'].isna().any():
        df_filtered['Type'].fillna('Power', inplace=True)
    
    # Fill missing voltage values with common defaults
    if 'Primary Voltage (kV)' in df_filtered.columns and df_filtered['Primary Voltage (kV)'].isna().any():
        df_filtered['Primary Voltage (kV)'].fillna(11.0, inplace=True)
    
    if 'Secondary Voltage (kV)' in df_filtered.columns and df_filtered['Secondary Voltage (kV)'].isna().any():
        df_filtered['Secondary Voltage (kV)'].fillna(0.433, inplace=True)
    
    # Log transformation for price (helps with skewed price distributions)
    df_filtered['price'] = np.log1p(df_filtered['price'])
    
    # Separate features and target
    X = df_filtered.drop('price', axis=1)
    y = df_filtered['price']
    
    # Double-check for missing values in X and y
    if X.isna().any().any():
        print("\nWarning: X still contains missing values after preprocessing:")
        for col in X.columns:
            missing = X[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing values")
    
    if y.isna().any():
        print(f"\nWarning: y contains {y.isna().sum()} missing values")
        print("Dropping these rows...")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("\nFeature types:")
    print(f"  Numerical features: {numerical_cols}")
    print(f"  Categorical features: {categorical_cols}")
    
    # Create preprocessing pipelines with imputers for missing values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nProcessed training data shape: {X_train_processed.shape}")
    print(f"Processed testing data shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, X.columns

def save_preprocessor(preprocessor):
    """Save the preprocessing pipeline"""
    filepath = 'models/preprocessor.joblib'
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to {filepath}")

def train_linear_model(X_train, y_train):
    """Train a Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_model(X_train, y_train, alpha=1.0):
    """Train a Ridge Regression model"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest Regressor"""
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """Train a Gradient Boosting Regressor"""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def optimize_hyperparameters(X_train, y_train, model_type='rf'):
    """Optimize hyperparameters using GridSearchCV"""
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        
    elif model_type == 'gb':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        model = GradientBoostingRegressor(random_state=42)
        
    elif model_type == 'ridge':
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        model = Ridge()
        
    elif model_type == 'lasso':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }
        model = Lasso()
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Return best model and parameters
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name=None, inverse_transform_fn=None):
    """
    Evaluate model performance
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test target values
    - model_name: Name of the model (for logging)
    - inverse_transform_fn: Function to convert log-transformed predictions back to original scale
    
    Returns:
    - Dictionary of evaluation metrics
    """
    model_name = model_name or model.__class__.__name__
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store original predictions and targets for visualization
    y_pred_orig = y_pred.copy()
    y_test_orig = y_test.copy()
    
    # If we used log transformation, convert back for proper metrics
    if inverse_transform_fn:
        y_pred = inverse_transform_fn(y_pred)
        y_test = inverse_transform_fn(y_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Return metrics and predictions
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred,
        'original_predictions': y_pred_orig,
        'original_targets': y_test_orig
    }

def save_model(model, model_name):
    """Save trained model to disk"""
    model_path = f'models/{model_name}.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def plot_predictions(y_test, y_pred, model_name, inverse_transform_fn=None):
    """
    Plot actual vs predicted values
    
    Parameters:
    - y_test: Actual values
    - y_pred: Predicted values
    - model_name: Name of the model
    - inverse_transform_fn: Function to convert log-transformed values back to original scale
    """
    # If we used log transformation, convert back for visualization
    if inverse_transform_fn:
        y_test_plot = inverse_transform_fn(y_test)
        y_pred_plot = inverse_transform_fn(y_pred)
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_plot, y_pred_plot, alpha=0.5)
    
    # Plot the identity line
    min_val = min(y_test_plot.min(), y_pred_plot.min())
    max_val = max(y_test_plot.max(), y_pred_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{model_name} - Actual vs Predicted Values')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create a DataFrame for easier sorting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.barh(range(len(feature_importance_df)), 
                feature_importance_df['Importance'], 
                align='center')
        plt.yticks(range(len(feature_importance_df)), 
                feature_importance_df['Feature'])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()
        
        # Return the DataFrame for further analysis
        return feature_importance_df
    else:
        print(f"Model {model_name} doesn't have feature_importances_ attribute")
        return None

def main():
    # Timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d')
    
    # Load the cleaned real transformer data
    data_path = f'data/transformer_real_data_cleaned_{timestamp}.csv'
    
    # Check if today's file exists, if not use the most recent
    if not os.path.exists(data_path):
        # Find the most recent cleaned data file
        data_files = [f for f in os.listdir('data') if f.startswith('transformer_real_data_cleaned_')]
        if data_files:
            # Sort by date in filename
            data_files.sort(reverse=True)
            data_path = os.path.join('data', data_files[0])
        else:
            print("No cleaned transformer data found. Please run the data cleaning script first.")
            return
    
    # Load data
    df = load_real_data(data_path)
    print(f"Loaded {len(df)} transformer records")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # Define inverse transform function for log-transformed prices
    inverse_transform_fn = lambda x: np.expm1(x)
    
    # Train and evaluate all models, storing results
    results = {}
    
    # Train and evaluate Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = train_linear_model(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression", inverse_transform_fn)
    results['Linear_Regression'] = lr_results
    save_model(lr_model, 'linear_regression')
    plot_predictions(y_test, lr_results['original_predictions'], "Linear Regression", inverse_transform_fn)
    
    # Train and evaluate Ridge Regression (with hyperparameter optimization)
    print("\nOptimizing Ridge Regression model...")
    ridge_model, ridge_params = optimize_hyperparameters(X_train, y_train, model_type='ridge')
    ridge_results = evaluate_model(ridge_model, X_test, y_test, "Ridge Regression", inverse_transform_fn)
    results['Ridge_Regression'] = ridge_results
    save_model(ridge_model, 'ridge_regression')
    plot_predictions(y_test, ridge_results['original_predictions'], "Ridge Regression", inverse_transform_fn)
    print(f"Best Ridge parameters: {ridge_params}")
    
    # Train and evaluate Random Forest (with hyperparameter optimization)
    print("\nOptimizing Random Forest model...")
    rf_model, rf_params = optimize_hyperparameters(X_train, y_train, model_type='rf')
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", inverse_transform_fn)
    results['Random_Forest'] = rf_results
    save_model(rf_model, 'random_forest')
    plot_predictions(y_test, rf_results['original_predictions'], "Random Forest", inverse_transform_fn)
    print(f"Best Random Forest parameters: {rf_params}")
    
    # Plot feature importance for Random Forest
    rf_feature_importance = plot_feature_importance(rf_model, feature_names, "Random Forest")
    if rf_feature_importance is not None:
        print("\nTop 10 features (Random Forest):")
        print(rf_feature_importance.head(10))
    
    # Train and evaluate Gradient Boosting (with hyperparameter optimization)
    print("\nOptimizing Gradient Boosting model...")
    gb_model, gb_params = optimize_hyperparameters(X_train, y_train, model_type='gb')
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", inverse_transform_fn)
    results['Gradient_Boosting'] = gb_results
    save_model(gb_model, 'gradient_boosting')
    plot_predictions(y_test, gb_results['original_predictions'], "Gradient Boosting", inverse_transform_fn)
    print(f"Best Gradient Boosting parameters: {gb_params}")
    
    # Plot feature importance for Gradient Boosting
    gb_feature_importance = plot_feature_importance(gb_model, feature_names, "Gradient Boosting")
    if gb_feature_importance is not None:
        print("\nTop 10 features (Gradient Boosting):")
        print(gb_feature_importance.head(10))
    
    # Print summary of results
    print("\n" + "="*50)
    print("Model Performance Summary (based on real market data):")
    print("="*50)
    
    # Sort models by R² score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    # Print results in tabular format
    print(f"{'Model':<20} {'RMSE ($)':<12} {'MAE ($)':<12} {'R²':<10} {'MAPE (%)':<10}")
    print("-"*64)
    for model_name, result in sorted_results:
        print(f"{model_name.replace('_', ' '):<20} {result['rmse']:<12.2f} {result['mae']:<12.2f} {result['r2']:<10.4f} {result['mape']:<10.2f}")
    
    # Identify best model
    best_model_name, best_model_results = sorted_results[0]
    print(f"\nBest model based on R²: {best_model_name.replace('_', ' ')} (R² = {best_model_results['r2']:.4f})")
    
    # Save model results summary
    results_df = pd.DataFrame({
        'Model': [model_name.replace('_', ' ') for model_name, _ in sorted_results],
        'RMSE': [result['rmse'] for _, result in sorted_results],
        'MAE': [result['mae'] for _, result in sorted_results],
        'R2': [result['r2'] for _, result in sorted_results],
        'MAPE': [result['mape'] for _, result in sorted_results]
    })
    
    results_df.to_csv(f'models/model_performance_summary_{timestamp}.csv', index=False)
    print(f"Results summary saved to models/model_performance_summary_{timestamp}.csv")

if __name__ == "__main__":
    main() 