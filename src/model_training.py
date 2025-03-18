import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler
import warnings

# Handle import either from src module or direct execution
try:
    from src.data_processing import load_data, preprocess_data, save_preprocessor, get_feature_names
except ModuleNotFoundError:
    from data_processing import load_data, preprocess_data, save_preprocessor, get_feature_names

# Create visualization directory
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Set Seaborn style
sns.set(style="whitegrid")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    mask = y_true != 0  # Avoid division by zero
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

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

def train_lasso_model(X_train, y_train, alpha=1.0):
    """Train a Lasso Regression model"""
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_elastic_net(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """Train an Elastic Net model"""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest Regressor"""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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

def train_svr(X_train, y_train, C=1.0, epsilon=0.1):
    """Train a Support Vector Regressor"""
    model = SVR(C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = calculate_mape(y_test, predictions)
    
    return {
        'predictions': predictions,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

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

def save_model(model, model_name):
    """Save trained model to disk"""
    model_path = f'../models/{model_name}_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.barh(range(min(20, len(indices))), 
                importances[indices][:20], 
                align='center')
        plt.yticks(range(min(20, len(indices))), 
                [feature_names[i] for i in indices][:20])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f'../visualizations/{model_name}_feature_importance.png')
        plt.close()

def plot_predictions(y_true, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Prices - {model_name}')
    plt.tight_layout()
    plt.savefig(f'../visualizations/{model_name}_predictions.png')
    plt.close()

def analyze_weight_scaling(df):
    """
    Analyze the relationship between power ratings and material weights
    """
    print("Analyzing weight scaling relationships...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('../visualizations/weight_analysis', exist_ok=True)
    
    # Analyze relationship between power rating and total weight
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='power_rating', y='total_weight', data=df)
    plt.title('Power Rating vs Total Weight')
    plt.xlabel('Power Rating (kVA)')
    plt.ylabel('Total Weight (kg)')
    plt.tight_layout()
    plt.savefig('../visualizations/weight_analysis/power_vs_total_weight.png')
    plt.close()
    
    # Analyze relationship between power rating and component weights
    weight_components = ['core_weight', 'copper_weight', 'insulation_weight', 'tank_weight', 'oil_weight']
    
    # Create subplots for each component
    fig, axes = plt.subplots(len(weight_components), 1, figsize=(12, 15))
    for i, component in enumerate(weight_components):
        sns.scatterplot(x='power_rating', y=component, data=df, ax=axes[i])
        axes[i].set_title(f'Power Rating vs {component.replace("_", " ").title()}')
        axes[i].set_xlabel('Power Rating (kVA)')
        axes[i].set_ylabel('Weight (kg)')
    
    plt.tight_layout()
    plt.savefig('../visualizations/weight_analysis/power_vs_component_weights.png')
    plt.close()
    
    # Calculate average weights by power rating
    weight_by_power = df.groupby('power_rating')[['total_weight'] + weight_components].mean().reset_index()
    
    # Plot average weights by power rating
    plt.figure(figsize=(12, 8))
    for component in weight_components:
        plt.plot(weight_by_power['power_rating'], weight_by_power[component], marker='o', label=component.replace('_', ' ').title())
    
    plt.xlabel('Power Rating (kVA)')
    plt.ylabel('Average Weight (kg)')
    plt.title('Average Component Weights by Power Rating')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../visualizations/weight_analysis/avg_weights_by_power.png')
    plt.close()
    
    # Calculate power-to-weight ratios
    df['power_to_weight_ratio'] = df['power_rating'] / df['total_weight']
    
    # Plot power-to-weight ratio
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='power_rating', y='power_to_weight_ratio', data=df)
    plt.title('Power-to-Weight Ratio by Power Rating')
    plt.xlabel('Power Rating (kVA)')
    plt.ylabel('Power-to-Weight Ratio (kVA/kg)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../visualizations/weight_analysis/power_to_weight_ratio.png')
    plt.close()
    
    # Save weight scaling data
    weight_by_power.to_csv('../visualizations/weight_analysis/weight_scaling_analysis.csv', index=False)
    
    return df

def add_weight_features(df):
    """
    Add weight-related features to improve model accuracy
    """
    # Calculate power-to-weight ratio
    df['power_to_weight_ratio'] = df['power_rating'] / df['total_weight']
    
    # Calculate component weight percentages
    weight_components = ['core_weight', 'copper_weight', 'insulation_weight', 'tank_weight', 'oil_weight']
    for component in weight_components:
        df[f'{component}_pct'] = df[component] / df['total_weight'] * 100
    
    # Calculate copper-to-core ratio (relevant for transformer design)
    df['copper_to_core_ratio'] = df['copper_weight'] / df['core_weight']
    
    # Calculate active part weight (core + copper + insulation)
    df['active_weight'] = df['core_weight'] + df['copper_weight'] + df['insulation_weight']
    df['active_weight_pct'] = df['active_weight'] / df['total_weight'] * 100
    
    # Calculate non-active part weight (tank + oil)
    df['non_active_weight'] = df['tank_weight'] + df['oil_weight']
    df['non_active_weight_pct'] = df['non_active_weight'] / df['total_weight'] * 100
    
    return df

def main():
    # Load data
    if os.path.exists('data/transformer_data.csv'):
        data_path = 'data/transformer_data.csv'
    else:
        data_path = '../data/transformer_data.csv'
    
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    # Analyze weight scaling
    df = analyze_weight_scaling(df)
    
    # Add weight-related features
    df = add_weight_features(df)
    print(f"Added weight-related features. Total features: {df.shape[1]}")
    
    # Save enhanced dataset
    enhanced_data_path = '../data/transformer_data_enhanced.csv'
    df.to_csv(enhanced_data_path, index=False)
    print(f"Enhanced dataset saved to {enhanced_data_path}")
    
    # Perform preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    feature_names = get_feature_names(preprocessor, df.drop('price', axis=1).columns)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # Initialize results dictionary
    results = {}
    
    # Train and evaluate Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = train_linear_model(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test)
    results['Linear_Regression'] = lr_results
    save_model(lr_model, 'linear_regression')
    plot_predictions(y_test, lr_results['predictions'], 'Linear_Regression')
    
    # Train and evaluate Ridge Regression
    print("\nOptimizing Ridge Regression model...")
    ridge_model, ridge_params = optimize_hyperparameters(X_train, y_train, model_type='ridge')
    ridge_results = evaluate_model(ridge_model, X_test, y_test)
    results['Ridge_Regression'] = ridge_results
    save_model(ridge_model, 'ridge_regression')
    plot_predictions(y_test, ridge_results['predictions'], 'Ridge_Regression')
    print(f"Best Ridge parameters: {ridge_params}")
    
    # Train and evaluate Lasso Regression
    print("\nOptimizing Lasso Regression model...")
    lasso_model, lasso_params = optimize_hyperparameters(X_train, y_train, model_type='lasso')
    lasso_results = evaluate_model(lasso_model, X_test, y_test)
    results['Lasso_Regression'] = lasso_results
    save_model(lasso_model, 'lasso_regression')
    plot_predictions(y_test, lasso_results['predictions'], 'Lasso_Regression')
    print(f"Best Lasso parameters: {lasso_params}")
    
    # Train and evaluate Random Forest
    print("\nOptimizing Random Forest model...")
    rf_model, rf_params = optimize_hyperparameters(X_train, y_train, model_type='rf')
    rf_results = evaluate_model(rf_model, X_test, y_test)
    results['Random_Forest'] = rf_results
    save_model(rf_model, 'random_forest')
    print(f"Best Random Forest parameters: {rf_params}")
    plot_feature_importance(rf_model, feature_names, 'Random_Forest')
    plot_predictions(y_test, rf_results['predictions'], 'Random_Forest')
    
    # Train and evaluate Gradient Boosting
    print("\nOptimizing Gradient Boosting model...")
    gb_model, gb_params = optimize_hyperparameters(X_train, y_train, model_type='gb')
    gb_results = evaluate_model(gb_model, X_test, y_test)
    results['Gradient_Boosting'] = gb_results
    save_model(gb_model, 'gradient_boosting')
    print(f"Best Gradient Boosting parameters: {gb_params}")
    plot_feature_importance(gb_model, feature_names, 'Gradient_Boosting')
    plot_predictions(y_test, gb_results['predictions'], 'Gradient_Boosting')
    
    # Print summary of results
    print("\n" + "="*50)
    print("Model Performance Summary:")
    print("="*50)
    summary_data = {
        'Model': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'MAPE (%)': []
    }
    
    for model_name, result in results.items():
        summary_data['Model'].append(model_name)
        summary_data['RMSE'].append(round(result['rmse'], 2))
        summary_data['MAE'].append(round(result['mae'], 2))
        summary_data['R²'].append(round(result['r2'], 4))
        summary_data['MAPE (%)'].append(round(result['mape'], 2))
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary results
    summary_df.to_csv('../models/model_performance_summary.csv', index=False)
    
    # Identify the best model based on R2 score
    best_model_name = summary_df.loc[summary_df['R²'].idxmax(), 'Model']
    print(f"\nBest performing model based on R² score: {best_model_name}")
    
    # Create a bar chart comparing model performance
    plt.figure(figsize=(12, 10))
    
    # RMSE subplot
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=summary_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('RMSE by Model (Lower is Better)')
    plt.tight_layout()
    
    # R² subplot
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='R²', data=summary_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('R² by Model (Higher is Better)')
    plt.tight_layout()
    
    # MAE subplot
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='MAE', data=summary_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('MAE by Model (Lower is Better)')
    plt.tight_layout()
    
    # MAPE subplot
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='MAPE (%)', data=summary_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('MAPE by Model (Lower is Better)')
    plt.tight_layout()
    
    plt.savefig('../visualizations/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes") 