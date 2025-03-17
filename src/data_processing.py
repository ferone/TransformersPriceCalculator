import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(file_path):
    """
    Load transformer data from CSV file
    """
    return pd.read_csv(file_path)

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess transformer data for machine learning
    
    Parameters:
    - df: DataFrame containing transformer data
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed training and testing data
    - preprocessor: Fitted preprocessor pipeline
    """
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
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
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def get_feature_names(preprocessor, input_features):
    """
    Get feature names after preprocessing
    """
    # Get names of one-hot encoded features
    cat_features = []
    for i, transformer_name in enumerate([name for name, _, _ in preprocessor.transformers_]):
        if transformer_name == 'cat':
            encoder = preprocessor.transformers_[i][1].named_steps['onehot']
            cat_cols = preprocessor.transformers_[i][2]
            for j, col in enumerate(cat_cols):
                categories = encoder.categories_[j]
                for cat in categories:
                    cat_features.append(f"{col}_{cat}")
    
    # Get names of numerical features
    num_features = [col for transformer_name, _, cols in preprocessor.transformers_ 
                   for col in cols if transformer_name == 'num']
    
    return num_features + cat_features

def save_preprocessor(preprocessor, filepath=None):
    """
    Save the preprocessing pipeline
    """
    if filepath is None:
        # Default path with flexibility for different calling contexts
        if os.path.exists('models'):
            filepath = 'models/preprocessor.joblib'
        else:
            filepath = '../models/preprocessor.joblib'
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    
if __name__ == "__main__":
    # Load data
    if os.path.exists('data/transformer_data.csv'):
        data_path = 'data/transformer_data.csv'
    else:
        data_path = '../data/transformer_data.csv'
        
    df = load_data(data_path)
    
    # Perform preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Save preprocessor for later use
    save_preprocessor(preprocessor)
    
    print(f"Data preprocessing complete")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, df.drop('price', axis=1).columns)
    print(f"Number of features after preprocessing: {len(feature_names)}") 