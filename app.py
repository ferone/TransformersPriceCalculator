import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from src.predict import predict_price, load_preprocessor, load_model

# Page configuration
st.set_page_config(
    page_title="Transformer Price Calculator",
    page_icon="⚡",
    layout="wide"
)

# Add requirements check for streamlit
try:
    import streamlit
except ImportError:
    print("Streamlit is not installed. Please install it using: pip install streamlit")

# Helper functions
def get_available_models():
    """Get list of available models"""
    # Check if models directory exists in current or parent directory
    if os.path.exists('models'):
        models_dir = 'models'
    else:
        models_dir = '../models'
        
    if not os.path.exists(models_dir):
        return []
    return [f.replace('.joblib', '') for f in os.listdir(models_dir) 
            if f.endswith('.joblib') and 'preprocessor' not in f]

def format_model_name(model_name):
    """Format model name for display"""
    return model_name.replace('_', ' ').title()

def display_model_performance():
    """Display model performance metrics if available"""
    # Check for performance file in current or parent directory
    if os.path.exists('models/model_performance_summary.csv'):
        performance_file = 'models/model_performance_summary.csv'
    else:
        performance_file = '../models/model_performance_summary.csv'
        
    if os.path.exists(performance_file):
        df = pd.read_csv(performance_file)
        
        # Display metrics table
        st.subheader("Model Performance Metrics")
        st.dataframe(df)
        
        # Plot performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RMSE subplot
        sns.barplot(x='Model', y='RMSE', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('RMSE by Model (Lower is Better)')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # R² subplot
        sns.barplot(x='Model', y='R²', data=df, ax=axes[0, 1])
        axes[0, 1].set_title('R² by Model (Higher is Better)')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # MAE subplot
        sns.barplot(x='Model', y='MAE', data=df, ax=axes[1, 0])
        axes[1, 0].set_title('MAE by Model (Lower is Better)')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # MAPE subplot
        sns.barplot(x='Model', y='MAPE (%)', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('MAPE by Model (Lower is Better)')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)

def display_feature_importance(model_name):
    """Display feature importance plot if available"""
    # Check visualizations directory in current or parent directory
    if os.path.exists('visualizations'):
        feature_imp_path = f'visualizations/{model_name}_feature_importance.png'
    else:
        feature_imp_path = f'../visualizations/{model_name}_feature_importance.png'
        
    if os.path.exists(feature_imp_path):
        st.subheader(f"Feature Importance for {format_model_name(model_name)}")
        image = Image.open(feature_imp_path)
        st.image(image, use_column_width=True)

def main():
    # App title and description
    st.title("⚡ Transformer Price Calculator")
    st.markdown("""
    Predict the price of electrical transformers based on specifications using machine learning models.
    
    Enter the transformer specifications below to get a price estimate.
    """)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This application uses machine learning models to predict the price of electrical transformers 
    based on various specifications such as power rating, material weights, and other parameters.
    
    Models available:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Elastic Net
    - Random Forest
    - Gradient Boosting
    
    The predictions are based on patterns learned from historical transformer pricing data.
    """)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Price Calculator", "Model Performance", "About Transformers"])
    
    with tab1:
        st.header("Transformer Specifications")
        
        # Model selection
        available_models = get_available_models()
        if not available_models:
            st.error("No trained models found. Please run the model training script first.")
            return
        
        selected_model = st.selectbox(
            "Select prediction model", 
            available_models,
            format_func=format_model_name,
            index=available_models.index('random_forest_regression') if 'random_forest_regression' in available_models else 0
        )
        
        # Create multicolumn layout for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Power and Voltage")
            power_rating = st.selectbox(
                "Power Rating (kVA)", 
                [25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000],
                index=6  # Default to 1000 kVA
            )
            
            primary_voltage = st.selectbox(
                "Primary Voltage (V)",
                [4160, 12470, 13200, 13800, 24940, 34500],
                index=3  # Default to 13800 V
            )
            
            secondary_voltage = st.selectbox(
                "Secondary Voltage (V)",
                [120, 208, 240, 277, 480, 600],
                index=4  # Default to 480 V
            )
            
            phase = st.radio(
                "Phase",
                ["Single-phase", "Three-phase"],
                index=1  # Default to three-phase
            )
            
            frequency = st.radio(
                "Frequency (Hz)",
                [50, 60],
                index=1  # Default to 60 Hz
            )
        
        with col2:
            st.subheader("Material Weights (kg)")
            core_weight = st.number_input("Core Weight", min_value=100.0, max_value=3000.0, value=1200.0, step=50.0)
            copper_weight = st.number_input("Copper Weight", min_value=50.0, max_value=2000.0, value=800.0, step=50.0)
            insulation_weight = st.number_input("Insulation Weight", min_value=20.0, max_value=500.0, value=200.0, step=10.0)
            tank_weight = st.number_input("Tank Weight", min_value=100.0, max_value=2500.0, value=950.0, step=50.0)
            oil_weight = st.number_input("Oil Weight", min_value=200.0, max_value=4000.0, value=1500.0, step=50.0)
            
            # Calculate total weight
            total_weight = core_weight + copper_weight + insulation_weight + tank_weight + oil_weight
            st.info(f"Total Weight: {total_weight:.2f} kg")
            
        with col3:
            st.subheader("Other Specifications")
            cooling_type = st.selectbox(
                "Cooling Type",
                ["ONAN", "ONAF", "OFAF", "ODAF"],
                index=1  # Default to ONAF
            )
            
            tap_changer = st.checkbox("Tap Changer", value=True)
            
            efficiency = st.slider(
                "Efficiency",
                min_value=0.95,
                max_value=0.99,
                value=0.975,
                step=0.001,
                format="%.3f"
            )
            
            impedance = st.slider(
                "Impedance (%)",
                min_value=2.0,
                max_value=8.0,
                value=5.5,
                step=0.1,
                format="%.1f"
            )
            
            installation_type = st.radio(
                "Installation Type",
                ["Indoor", "Outdoor"],
                index=1  # Default to Outdoor
            )
            
            insulation_type = st.radio(
                "Insulation Type",
                ["Oil", "Dry"],
                index=0  # Default to Oil
            )
            
            manufacturing_location = st.selectbox(
                "Manufacturing Location",
                ["North America", "Europe", "Asia", "South America"],
                index=0  # Default to North America
            )
        
        # Create specifications dictionary
        transformer_specs = {
            'power_rating': power_rating,
            'primary_voltage': primary_voltage,
            'secondary_voltage': secondary_voltage,
            'core_weight': core_weight,
            'copper_weight': copper_weight,
            'insulation_weight': insulation_weight,
            'tank_weight': tank_weight,
            'oil_weight': oil_weight,
            'total_weight': total_weight,
            'cooling_type': cooling_type,
            'tap_changer': tap_changer,
            'efficiency': efficiency,
            'impedance': impedance,
            'phase': phase,
            'frequency': frequency,
            'installation_type': installation_type,
            'insulation_type': insulation_type,
            'manufacturing_location': manufacturing_location
        }
        
        # Make predictions when button is clicked
        if st.button("Calculate Transformer Price"):
            try:
                # Load preprocessor and model
                preprocessor = load_preprocessor()
                
                # Check model path in both possible locations
                if os.path.exists(f"models/{selected_model}.joblib"):
                    model_path = f"models/{selected_model}.joblib"
                else:
                    model_path = f"../models/{selected_model}.joblib"
                    
                model = load_model(model_path)
                
                # Make prediction
                predicted_price = predict_price(model, preprocessor, transformer_specs)
                
                # Display result
                st.success(f"## Estimated Price: ${predicted_price:,.2f}")
                
                # Show feature importance if available
                display_feature_importance(selected_model)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    with tab2:
        st.header("Model Performance Analysis")
        display_model_performance()
    
    with tab3:
        st.header("About Electrical Transformers")
        st.markdown("""
        ### What are Electrical Transformers?
        Electrical transformers are static electrical devices that transfer electrical energy between two or more circuits through electromagnetic induction. 
        They are used to increase or decrease the voltage in power transmission and distribution systems.

        ### Key Specifications:
        
        - **Power Rating (kVA):** The capacity of the transformer to deliver power.
        - **Voltage Ratings:** Primary and secondary voltage levels.
        - **Phase:** Single-phase or three-phase configuration.
        - **Material Weights:** Core, copper, insulation, tank, and oil weights affect cost significantly.
        - **Cooling Type:**
          - **ONAN:** Oil Natural, Air Natural cooling
          - **ONAF:** Oil Natural, Air Forced cooling
          - **OFAF:** Oil Forced, Air Forced cooling
          - **ODAF:** Oil Directed, Air Forced cooling
        - **Tap Changer:** Device for voltage regulation, adds to transformer cost.
        - **Efficiency:** Higher efficiency transformers cost more but have lower operating costs.
        - **Impedance:** Affects short-circuit behavior and system stability.
        
        ### Factors Affecting Transformer Prices:
        
        1. **Raw Material Costs:** Copper, steel, and insulation material prices
        2. **Power Rating:** Higher rated transformers cost more
        3. **Design Complexity:** Special features add to cost
        4. **Manufacturing Location:** Labor and manufacturing costs vary by region
        5. **Market Conditions:** Supply and demand dynamics
        6. **Transportation Cost:** Especially significant for large transformers
        """)

if __name__ == "__main__":
    main() 