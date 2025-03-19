"""
Transformer Bridge Module

This module serves as a bridge between the Streamlit application and 
the trained transformer models, handling field name conversions and data validation.
"""

import os
import joblib
import pandas as pd
import numpy as np

def map_fields(app_specs):
    """
    Maps field names from app format to model format
    
    Parameters:
    - app_specs: Dictionary with app-formatted transformer specifications
    
    Returns:
    - Dictionary with model-formatted transformer specifications
    """
    # Create field mapping (app field name -> model field name)
    field_mapping = {
        'power_rating': 'Power Rating (KVA)',
        'primary_voltage': 'Primary Voltage (kV)',
        'secondary_voltage': 'Secondary Voltage (kV)',
        'phase': 'Phase',
        'installation_type': 'Type',
        'cooling_type': 'Cooling Type',
        'frequency': 'Frequency (Hz)',
        'tap_changer': 'Tap Changer'
    }
    
    # Create a new dictionary with mapped field names
    model_specs = {}
    for app_field, model_field in field_mapping.items():
        if app_field in app_specs:
            model_specs[model_field] = app_specs[app_field]
    
    # Handle special cases or transformations if needed
    if 'Type' not in model_specs and 'installation_type' in app_specs:
        # Map installation type to Type field
        installation_map = {
            'Indoor': 'Power',
            'Outdoor': 'Power',
            'Pad Mounted': 'Distribution',
            'Pole Mounted': 'Distribution'
        }
        model_specs['Type'] = installation_map.get(app_specs['installation_type'], 'Power')
    
    # Ensure Phase is formatted correctly
    if 'Phase' in model_specs:
        # Make sure phase is properly formatted
        if model_specs['Phase'].lower() in ['three', '3', 'three phase', '3-phase']:
            model_specs['Phase'] = 'Three-phase'
        elif model_specs['Phase'].lower() in ['single', '1', 'single phase', '1-phase']:
            model_specs['Phase'] = 'Single-phase'
    
    return model_specs

def validate_model_specs(model_specs):
    """
    Validates the model specifications and attempts to fix common issues
    
    Parameters:
    - model_specs: Dictionary with model-formatted transformer specifications
    
    Returns:
    - Validated and corrected model specifications
    - List of warnings or notes about changes made
    """
    validated_specs = model_specs.copy()
    warnings = []
    
    # Required fields for the model
    required_fields = [
        'Power Rating (KVA)', 
        'Primary Voltage (kV)', 
        'Secondary Voltage (kV)', 
        'Phase', 
        'Type'
    ]
    
    # Check for missing required fields and provide defaults
    for field in required_fields:
        if field not in validated_specs or validated_specs[field] is None:
            if field == 'Power Rating (KVA)':
                validated_specs[field] = 1000.0
                warnings.append(f"Missing {field}: Using default value of 1000.0 KVA")
            elif field == 'Primary Voltage (kV)':
                validated_specs[field] = 11.0
                warnings.append(f"Missing {field}: Using default value of 11.0 kV")
            elif field == 'Secondary Voltage (kV)':
                validated_specs[field] = 0.433
                warnings.append(f"Missing {field}: Using default value of 0.433 kV")
            elif field == 'Phase':
                validated_specs[field] = 'Three-phase'
                warnings.append(f"Missing {field}: Using default value of Three-phase")
            elif field == 'Type':
                validated_specs[field] = 'Power'
                warnings.append(f"Missing {field}: Using default value of Power")
    
    # Ensure numeric fields are really numeric
    numeric_fields = ['Power Rating (KVA)', 'Primary Voltage (kV)', 'Secondary Voltage (kV)']
    for field in numeric_fields:
        if field in validated_specs:
            try:
                value = validated_specs[field]
                if isinstance(value, str):
                    # Try to convert to float, removing any units or commas
                    cleaned_value = value.replace(',', '').replace('KVA', '').replace('kV', '').strip()
                    validated_specs[field] = float(cleaned_value)
                    warnings.append(f"Converted {field} from '{value}' to {validated_specs[field]}")
            except (ValueError, TypeError):
                # If conversion fails, use default values
                if field == 'Power Rating (KVA)':
                    validated_specs[field] = 1000.0
                    warnings.append(f"Invalid {field} value '{validated_specs[field]}': Using default of 1000.0 KVA")
                elif field == 'Primary Voltage (kV)':
                    validated_specs[field] = 11.0
                    warnings.append(f"Invalid {field} value '{validated_specs[field]}': Using default of 11.0 kV")
                elif field == 'Secondary Voltage (kV)':
                    validated_specs[field] = 0.433
                    warnings.append(f"Invalid {field} value '{validated_specs[field]}': Using default of 0.433 kV")
    
    return validated_specs, warnings

def predict_transformer_price(app_specs, model_name='random_forest'):
    """
    Predict transformer price using app specifications
    
    Parameters:
    - app_specs: Dictionary with app-formatted transformer specifications
    - model_name: Name of the model to use for prediction
    
    Returns:
    - predicted_price: The predicted price
    - warnings: List of warnings or notes about the prediction
    """
    warnings = []
    
    # Map app fields to model fields
    try:
        model_specs = map_fields(app_specs)
        model_specs, field_warnings = validate_model_specs(model_specs)
        warnings.extend(field_warnings)
    except Exception as e:
        raise ValueError(f"Error in field mapping: {str(e)}")
    
    # Load the model
    model_path = f'models/{model_name}.joblib'
    if not os.path.exists(model_path):
        model_path = f'../models/{model_name}.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_name}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")
    
    # Try to load preprocessor
    preprocessor = None
    preprocessor_path = 'models/preprocessor.joblib'
    if not os.path.exists(preprocessor_path):
        preprocessor_path = '../models/preprocessor.joblib'
    
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            warnings.append(f"Could not load preprocessor: {str(e)}. Will attempt direct prediction.")
    
    # Check if we need to use a simplified approach based on model type
    is_linear_model = any(x in model_name.lower() for x in ['linear', 'ridge', 'lasso', 'elasticnet'])
    
    # Make prediction
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([model_specs])
        
        # Separate numeric and categorical fields to handle them properly
        # Known categorical fields
        categorical_fields = ['Phase', 'Type', 'Cooling Type']
        for field in categorical_fields:
            if field in input_df.columns:
                # Ensure the field is treated as a categorical/object type
                input_df[field] = input_df[field].astype(str)
        
        # Try to use preprocessor first
        if preprocessor is not None:
            try:
                # Print debug info
                warnings.append(f"Input data columns: {input_df.columns.tolist()}")
                
                # Transform the input data
                input_processed = preprocessor.transform(input_df)
                
                # Make prediction (our model was trained on log-transformed prices)
                log_price = model.predict(input_processed)[0]
                
                # Convert back from log scale
                predicted_price = np.expm1(log_price)
                return predicted_price, warnings
            except Exception as e:
                warnings.append(f"Preprocessor failed: {str(e)}. Attempting direct prediction.")
        
        # For linear models which are strict about feature count, use a more sophisticated approach
        if is_linear_model:
            # Check if we have information about the expected features
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
            if expected_features is not None and len(expected_features) > 0:
                warnings.append(f"Using custom approach for {model_name} model with {len(expected_features)} expected features")
                
                # Create a DataFrame with expected feature names, filled with zeros
                dummy_df = pd.DataFrame(0, index=[0], columns=expected_features)
                
                # Set reasonable values for common features - this is just an estimation
                # Power Rating is the most important feature
                if 'Power Rating (KVA)' in input_df.columns and any('power' in feat.lower() for feat in expected_features):
                    power = float(input_df['Power Rating (KVA)'].iloc[0])
                    power_features = [feat for feat in expected_features if 'power' in feat.lower()]
                    if power_features:
                        dummy_df[power_features[0]] = power
                
                # Try to predict with the dummy features
                try:
                    log_price = model.predict(dummy_df)[0]
                    # Convert back from log scale if needed
                    if log_price < 100:  # Assume low values are log-transformed
                        predicted_price = np.expm1(log_price)
                    else:
                        predicted_price = log_price
                    
                    warnings.append(f"Used dummy feature matrix for {model_name}. Results may be less accurate.")
                    return predicted_price, warnings
                except Exception as dummy_error:
                    warnings.append(f"Dummy feature prediction failed: {str(dummy_error)}")
            
            # If dummy approach failed or not applicable, use enhanced power-based formula
            warnings.append(f"Using enhanced power-based formula for {model_name} model")
            
            # Get key specifications
            power = 1000.0  # default
            if 'Power Rating (KVA)' in input_df.columns:
                power = float(input_df['Power Rating (KVA)'].iloc[0])
                
            primary_voltage = 11.0  # default
            if 'Primary Voltage (kV)' in input_df.columns:
                primary_voltage = float(input_df['Primary Voltage (kV)'].iloc[0])
                
            secondary_voltage = 0.433  # default
            if 'Secondary Voltage (kV)' in input_df.columns:
                secondary_voltage = float(input_df['Secondary Voltage (kV)'].iloc[0])
            
            # Enhanced pricing formula based on industry knowledge
            # Base price depends on power rating
            if power <= 500:
                base_price = 5000 + (power * 25)  # Small transformers: higher $ per KVA
            elif power <= 2000:
                base_price = 10000 + (power * 20)  # Medium transformers
            elif power <= 10000:
                base_price = 25000 + (power * 15)  # Large transformers: economies of scale
            else:
                base_price = 75000 + (power * 10)  # Very large transformers: more economies of scale
            
            # Adjust for voltage ratings
            # Higher voltage ratings increase price
            voltage_factor = 1.0
            if primary_voltage > 33:
                voltage_factor += 0.1 * (primary_voltage / 33 - 1)  # +10% for each 33kV above 33kV
            
            # Large voltage transformation ratio increases complexity and cost
            ratio = primary_voltage / secondary_voltage if secondary_voltage > 0 else 1
            if ratio > 10:
                voltage_factor += 0.05 * (ratio / 10 - 1)  # +5% for each 10x ratio above 10x
            
            # Cooling type factor (if available)
            cooling_factor = 1.0
            if 'Cooling Type' in input_df.columns:
                cooling_type = input_df['Cooling Type'].iloc[0]
                if cooling_type == 'ONAF':
                    cooling_factor = 1.1  # +10% for forced air cooling
                elif cooling_type == 'OFAF':
                    cooling_factor = 1.2  # +20% for forced oil and air cooling
                elif cooling_type == 'ODAF':
                    cooling_factor = 1.3  # +30% for directed oil flow and forced air
            
            # Phase factor
            phase_factor = 1.0
            if 'Phase' in input_df.columns:
                phase = input_df['Phase'].iloc[0]
                if phase == 'Single-phase':
                    phase_factor = 0.7  # Single-phase transformers are generally less expensive
            
            # Calculate final price
            predicted_price = base_price * voltage_factor * cooling_factor * phase_factor
            
            warnings.append(f"Used enhanced power-based estimation: ${predicted_price:.2f}")
            warnings.append("Note: This is an approximation using industry price factors.")
            
            return predicted_price, warnings
        
        # If it's a tree-based model (like Random Forest or Gradient Boosting),
        # we have more flexibility with features
        try:
            # Check for expected features
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                warnings.append(f"Model expects {len(expected_features)} features")
            
            # If we know what features are expected, try to match them
            if expected_features is not None:
                # Manually encode categorical variables
                encoded_df = input_df.copy()
                for field in categorical_fields:
                    if field in encoded_df.columns:
                        # Use simple one-hot encoding
                        dummies = pd.get_dummies(encoded_df[field], prefix=field)
                        encoded_df = pd.concat([encoded_df.drop(field, axis=1), dummies], axis=1)
                
                # Create a DataFrame with expected feature names, filled with zeros
                expected_df = pd.DataFrame(0, index=[0], columns=expected_features)
                
                # Copy available features to the expected DataFrame
                for col in encoded_df.columns:
                    if col in expected_features:
                        expected_df[col] = encoded_df[col]
                
                # For categorical one-hot encoded columns, try to match partially
                for col in expected_features:
                    for prefix in categorical_fields:
                        # If the column name has a prefix from categorical fields
                        if col.startswith(f"{prefix}_") or col.startswith(f"{prefix}."):
                            # Try to find a match in the input data
                            suffix = col.split('_', 1)[1] if '_' in col else col.split('.', 1)[1]
                            for field in categorical_fields:
                                if field in input_df.columns:
                                    # If the categorical value matches the suffix
                                    if input_df[field].iloc[0].replace(' ', '_').lower() == suffix.lower():
                                        expected_df[col] = 1
                                        warnings.append(f"Matched categorical feature {col}")
                                
                # Make prediction
                warnings.append(f"Using expected feature matrix with {len(expected_features)} columns")
                log_price = model.predict(expected_df)[0]
                
                # Convert back from log scale if needed
                if log_price < 100:  # Assume low values are log-transformed
                    predicted_price = np.expm1(log_price)
                else:
                    predicted_price = log_price
                
                return predicted_price, warnings
            
            # If we don't know expected features, fallback to only numeric features
            numeric_df = input_df.select_dtypes(include=['number'])
            predicted_price = model.predict(numeric_df)[0]
            if predicted_price < 100:  # Assume low values are log-transformed
                predicted_price = np.expm1(predicted_price)
            
            return predicted_price, warnings
            
        except Exception as e:
            # Final fallback: enhanced power-based estimation
            warnings.append(f"All prediction attempts failed: {str(e)}. Using enhanced estimation.")
            
            # Get key specifications
            power = 1000.0  # default
            if 'Power Rating (KVA)' in input_df.columns:
                power = float(input_df['Power Rating (KVA)'].iloc[0])
                
            primary_voltage = 11.0  # default
            if 'Primary Voltage (kV)' in input_df.columns:
                primary_voltage = float(input_df['Primary Voltage (kV)'].iloc[0])
                
            secondary_voltage = 0.433  # default
            if 'Secondary Voltage (kV)' in input_df.columns:
                secondary_voltage = float(input_df['Secondary Voltage (kV)'].iloc[0])
            
            # Enhanced pricing formula
            # Base price depends on power rating
            if power <= 500:
                base_price = 5000 + (power * 25)  # Small transformers: higher $ per KVA
            elif power <= 2000:
                base_price = 10000 + (power * 20)  # Medium transformers
            elif power <= 10000:
                base_price = 25000 + (power * 15)  # Large transformers: economies of scale
            else:
                base_price = 75000 + (power * 10)  # Very large transformers: more economies of scale
            
            # Adjust for voltage ratings
            voltage_factor = 1.0
            if primary_voltage > 33:
                voltage_factor += 0.1 * (primary_voltage / 33 - 1)
            
            ratio = primary_voltage / secondary_voltage if secondary_voltage > 0 else 1
            if ratio > 10:
                voltage_factor += 0.05 * (ratio / 10 - 1)
            
            # Cooling type factor
            cooling_factor = 1.0
            if 'Cooling Type' in input_df.columns:
                cooling_type = input_df['Cooling Type'].iloc[0]
                if cooling_type == 'ONAF':
                    cooling_factor = 1.1
                elif cooling_type == 'OFAF':
                    cooling_factor = 1.2
                elif cooling_type == 'ODAF':
                    cooling_factor = 1.3
            
            # Phase factor
            phase_factor = 1.0
            if 'Phase' in input_df.columns:
                phase = input_df['Phase'].iloc[0]
                if phase == 'Single-phase':
                    phase_factor = 0.7
            
            # Calculate final price
            predicted_price = base_price * voltage_factor * cooling_factor * phase_factor
            
            warnings.append(f"Used enhanced industry-based estimation: ${predicted_price:.2f}")
            return predicted_price, warnings
        
    except Exception as e:
        raise ValueError(f"Error predicting price: {str(e)}")

def get_all_model_predictions(app_specs):
    """
    Get predictions from all available models
    
    Parameters:
    - app_specs: Dictionary with app-formatted transformer specifications
    
    Returns:
    - Dictionary of model name -> (predicted_price, warnings)
    """
    results = {}
    
    # Map app fields to model fields
    try:
        model_specs = map_fields(app_specs)
        model_specs, global_warnings = validate_model_specs(model_specs)
    except Exception as e:
        raise ValueError(f"Error in field mapping: {str(e)}")
    
    # Get available models
    if os.path.exists('models'):
        models_dir = 'models'
    else:
        models_dir = '../models'
        
    if not os.path.exists(models_dir):
        return {}
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and not f.startswith('preprocessor')]
    
    # Try to load preprocessor
    preprocessor = None
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception:
            pass
    
    # Get predictions from each model
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        model_display_name = model_name.replace('_', ' ').title()
        
        try:
            # Get prediction for this model
            predicted_price, warnings = predict_transformer_price(app_specs, model_name)
            
            # Add warnings about model type
            if any(x in model_name.lower() for x in ['linear', 'ridge', 'lasso', 'elasticnet']):
                warnings.append("Linear model predictions may use fallback estimation due to feature compatibility issues.")
                warnings.append("For most accurate results, use tree-based models like Random Forest or Gradient Boosting.")
            
            # Add the prediction to results
            results[model_display_name] = (predicted_price, warnings)
        except Exception as e:
            results[model_display_name] = (None, [f"Prediction failed: {str(e)}"])
    
    return results 