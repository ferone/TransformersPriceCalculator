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

# Handle import either from src module or direct execution
try:
    from src.data_processing import load_data, preprocess_data, save_preprocessor, get_feature_names
except ModuleNotFoundError:
    from data_processing import load_data, preprocess_data, save_preprocessor, get_feature_names

def train_linear_model(X_train, y_train):
    """Train a linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_model(X_train, y_train, alpha=1.0):
    """Train a ridge regression model"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_model(X_train, y_train, alpha=1.0):
    """Train a lasso regression model"""
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_elastic_net(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """Train an elastic net regression model"""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a random forest regression model"""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """Train a gradient boosting regression model"""
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train, C=1.0, kernel='rbf'):
    """Train a support vector regression model"""
    model = SVR(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate regression model performance"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred
    }

def optimize_hyperparameters(X_train, y_train, model_type='rf'):
    """Optimize hyperparameters for the selected model type"""
    if model_type == 'ridge':
        model = Ridge()
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model_type == 'lasso':
        model = Lasso()
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    elif model_type == 'elastic_net':
        model = ElasticNet()
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    elif model_type == 'svr':
        model = SVR()
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, model_name, directory='../models'):
    """Save trained model to file"""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, f"{directory}/{model_name}.joblib")

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based models"""
    # Check if model has feature importances
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('../visualizations', exist_ok=True)
        
        # Save plot
        plt.savefig(f'../visualizations/{model_name}_feature_importance.png')
        plt.close()

def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Transformer Prices - {model_name}')
    
    # Create directory if it doesn't exist
    os.makedirs('../visualizations', exist_ok=True)
    
    # Save plot
    plt.savefig(f'../visualizations/{model_name}_actual_vs_predicted.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_data('../data/transformer_data.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(preprocessor, data.drop('price', axis=1).columns)
    
    # Dictionary to store model results
    results = {}
    
    # Train and evaluate Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = train_linear_model(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test)
    save_model(lr_model, 'linear_regression')
    results['Linear Regression'] = lr_results
    plot_predictions(y_test, lr_results['predictions'], 'Linear_Regression')
    
    # Train and evaluate Ridge Regression with hyperparameter optimization
    print("\nOptimizing Ridge Regression...")
    ridge_model, ridge_params = optimize_hyperparameters(X_train, y_train, 'ridge')
    ridge_results = evaluate_model(ridge_model, X_test, y_test)
    save_model(ridge_model, 'ridge_regression')
    results['Ridge Regression'] = ridge_results
    print(f"Best Ridge parameters: {ridge_params}")
    plot_predictions(y_test, ridge_results['predictions'], 'Ridge_Regression')
    
    # Train and evaluate Lasso Regression with hyperparameter optimization
    print("\nOptimizing Lasso Regression...")
    lasso_model, lasso_params = optimize_hyperparameters(X_train, y_train, 'lasso')
    lasso_results = evaluate_model(lasso_model, X_test, y_test)
    save_model(lasso_model, 'lasso_regression')
    results['Lasso Regression'] = lasso_results
    print(f"Best Lasso parameters: {lasso_params}")
    plot_predictions(y_test, lasso_results['predictions'], 'Lasso_Regression')
    
    # Train and evaluate Elastic Net with hyperparameter optimization
    print("\nOptimizing Elastic Net Regression...")
    elastic_model, elastic_params = optimize_hyperparameters(X_train, y_train, 'elastic_net')
    elastic_results = evaluate_model(elastic_model, X_test, y_test)
    save_model(elastic_model, 'elastic_net_regression')
    results['Elastic Net'] = elastic_results
    print(f"Best Elastic Net parameters: {elastic_params}")
    plot_predictions(y_test, elastic_results['predictions'], 'Elastic_Net')
    
    # Train and evaluate Random Forest with hyperparameter optimization
    print("\nOptimizing Random Forest Regression...")
    rf_model, rf_params = optimize_hyperparameters(X_train, y_train, 'rf')
    rf_results = evaluate_model(rf_model, X_test, y_test)
    save_model(rf_model, 'random_forest_regression')
    results['Random Forest'] = rf_results
    print(f"Best Random Forest parameters: {rf_params}")
    plot_feature_importance(rf_model, feature_names, 'Random_Forest')
    plot_predictions(y_test, rf_results['predictions'], 'Random_Forest')
    
    # Train and evaluate Gradient Boosting with hyperparameter optimization
    print("\nOptimizing Gradient Boosting Regression...")
    gb_model, gb_params = optimize_hyperparameters(X_train, y_train, 'gb')
    gb_results = evaluate_model(gb_model, X_test, y_test)
    save_model(gb_model, 'gradient_boosting_regression')
    results['Gradient Boosting'] = gb_results
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