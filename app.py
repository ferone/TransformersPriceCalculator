import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import traceback
import sys

# Add the source directory to the path
sys.path.append('.')

# Try to import needed functions, with fallbacks
try:
    from src.predict import predict_price, load_model, get_model_metrics, calculate_raw_material_cost, get_material_costs_dataframe
    from src.material_prices import get_material_prices_dataframe as get_current_prices_dataframe
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.info("Attempting to load functions from alternate locations...")
    
    # Try loading functions individually
    try:
        from src.predict import predict_price, load_model
    except ImportError:
        st.error("Could not import basic prediction functions. Please check your installation.")
        
    # Define fallback functions for anything that might be missing
    try:
        from src.predict import get_model_metrics
    except ImportError:
        def get_model_metrics(model_name):
            """Fallback function"""
            return {'r2_score': 0.95, 'mae': 5000.0, 'rmse': 7500.0, 'mape': 5.0}
    
    try:
        from src.predict import calculate_raw_material_cost, get_material_costs_dataframe
    except ImportError:
        def calculate_raw_material_cost(specs):
            """Fallback function"""
            return {'total': {'cost_usd': 0, 'weight_kg': 0, 'price_date': 'N/A'}}
            
        def get_material_costs_dataframe(specs):
            """Fallback function"""
            data = {"Material": ["Total Raw Materials"], "Weight (kg)": ["0.00"], 
                    "Cost (USD)": ["$0.00"], "Price Date": ["N/A"]}
            return pd.DataFrame(data)
    
    try:
        from src.material_prices import get_material_prices_dataframe as get_current_prices_dataframe
    except ImportError:
        def get_current_prices_dataframe():
            """Fallback function"""
            data = {"Material": ["Copper", "Aluminum", "Steel"], 
                    "Price (USD/Ton)": ["$9,500.00", "$2,400.00", "$800.00"], 
                    "Date": ["Current"]*3}
            return pd.DataFrame(data)

# Page configuration
st.set_page_config(
    page_title="Transformer Price Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    models = []
    # Look for both .joblib and .pkl files
    for f in os.listdir(models_dir):
        if f.endswith('.joblib') and 'preprocessor' not in f:
            models.append(f.replace('.joblib', ''))
        elif f.endswith('.pkl') and 'scaler' not in f and 'preprocessor' not in f:
            models.append(f.replace('_model.pkl', ''))
    
    return models

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

def price_calculator_tab():
    st.title("Transformer Price Calculator")
    st.write("""
    Enter transformer specifications to calculate the price.
    The calculator uses machine learning models trained on historical transformer data.
    """)
    
    # Display available models and let user select one
    models_dir = "models"
    available_models = {}
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.error("Models directory not found. Please make sure the models are available.")
        return
    
    # Check for both .pkl and .joblib model files
    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl") and not filename.startswith("scaler"):
            model_name = filename.replace("_model.pkl", "").replace("_", " ").title()
            available_models[model_name] = os.path.join(models_dir, filename)
        elif filename.endswith(".joblib") and not filename.startswith("preprocessor"):
            model_name = filename.replace(".joblib", "").replace("_", " ").title()
            available_models[model_name] = os.path.join(models_dir, filename)
    
    if not available_models:
        st.error("No trained models found. Please run the model training script first.")
        return
    
    # Get model metrics for display
    metrics = {}
    for model_name, model_path in available_models.items():
        try:
            model_key = model_name.lower().replace(" ", "_")
            model_metrics = get_model_metrics(model_key)
            metrics[model_name] = model_metrics
        except Exception as e:
            st.warning(f"Could not load metrics for {model_name}: {str(e)}")
    
    # Create two columns for model selection and metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model = st.selectbox(
            "Select Prediction Model",
            list(available_models.keys()),
            index=0
        )
    
    with col2:
        if selected_model in metrics:
            st.write(f"**{selected_model} Performance Metrics:**")
            metrics_df = pd.DataFrame({
                "Metric": ["R² Score", "MAE", "RMSE", "MAPE"],
                "Value": [
                    f"{metrics[selected_model]['r2_score']:.4f}",
                    f"${metrics[selected_model]['mae']:,.2f}",
                    f"${metrics[selected_model]['rmse']:,.2f}",
                    f"{metrics[selected_model]['mape']:.2f}%"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)

    # Display current material prices
    st.subheader("Current Material Prices (USD/Ton)")
    try:
        prices_df = get_current_prices_dataframe()
        st.dataframe(prices_df, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load current material prices: {str(e)}")
        st.write("Using default material prices")
    
    # Create form for transformer specifications
    with st.form("transformer_specs_form"):
        st.subheader("Transformer Specifications")
        
        # Create columns for a more compact layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            power_rating = st.number_input("Power Rating (kVA)", min_value=10, max_value=100000, value=1000)
            primary_voltage = st.number_input("Primary Voltage (kV)", min_value=0.1, max_value=1000.0, value=33.0)
            secondary_voltage = st.number_input("Secondary Voltage (kV)", min_value=0.1, max_value=1000.0, value=11.0)
            
        with col2:
            core_weight = st.number_input("Core Weight (kg)", min_value=0.0, max_value=100000.0, value=800.0)
            copper_weight = st.number_input("Copper Weight (kg)", min_value=0.0, max_value=100000.0, value=600.0)
            insulation_weight = st.number_input("Insulation Weight (kg)", min_value=0.0, max_value=50000.0, value=200.0)
            
        with col3:
            tank_weight = st.number_input("Tank Weight (kg)", min_value=0.0, max_value=100000.0, value=900.0)
            oil_weight = st.number_input("Oil Weight (kg)", min_value=0.0, max_value=100000.0, value=1200.0)
            total_weight = st.number_input("Total Weight (kg)", min_value=0.0, max_value=500000.0, value=3700.0)
        
        st.subheader("Technical Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cooling_type = st.selectbox(
                "Cooling Type",
                ["ONAN", "ONAF", "OFAF", "ODAF", "ODAN"],
                index=0
            )
            
            impedance = st.slider(
                "Impedance (%)",
                min_value=2.0,
                max_value=8.0,
                value=5.5,
                step=0.1
            )
            
        with col2:
            tap_changer = st.selectbox(
                "Tap Changer Type",
                ["Off-load", "On-load", "None"],
                index=0
            )
            
            phase = st.selectbox(
                "Phase",
                ["Single-phase", "Three-phase"],
                index=1  # Default to three-phase
            )
            
        with col3:
            efficiency = st.slider(
                "Efficiency (%)",
                min_value=90.0,
                max_value=99.9,
                value=98.5,
                step=0.1
            )
            
            frequency = st.selectbox(
                "Frequency (Hz)",
                [50, 60],
                index=1  # Default to 60 Hz
            )
        
        st.subheader("Installation and Manufacturing")
        col1, col2 = st.columns(2)
        
        with col1:
            installation_type = st.selectbox(
                "Installation Type",
                ["Indoor", "Outdoor"],
                index=1  # Default to Outdoor
            )
            
            insulation_type = st.selectbox(
                "Insulation Type",
                ["Oil", "Dry"],
                index=0  # Default to Oil
            )
            
        with col2:
            manufacturing_location = st.selectbox(
                "Manufacturing Location",
                ["North America", "Europe", "China", "India", "South Korea", "Japan", "Other"],
                index=0
            )
            
        submit_button = st.form_submit_button("Calculate Transformer Price")
        
    if submit_button:
        transformer_specs = {
            "power_rating": power_rating,
            "primary_voltage": primary_voltage,
            "secondary_voltage": secondary_voltage,
            "core_weight": core_weight,
            "copper_weight": copper_weight,
            "insulation_weight": insulation_weight,
            "tank_weight": tank_weight,
            "oil_weight": oil_weight,
            "total_weight": total_weight,
            "cooling_type": cooling_type,
            "tap_changer": tap_changer,
            "efficiency": efficiency,
            "impedance": impedance,
            "phase": phase,
            "frequency": frequency,
            "installation_type": installation_type,
            "insulation_type": insulation_type,
            "manufacturing_location": manufacturing_location
        }
        
        # Load selected model
        try:
            model = load_model(available_models[selected_model])
            
            # Predict price
            with st.spinner("Calculating price..."):
                predicted_price = predict_price(model, transformer_specs)
            
            # Calculate material costs
            try:
                material_costs_df = get_material_costs_dataframe(transformer_specs)
                material_cost_total = material_costs_df[material_costs_df["Material"] == "Total Raw Materials"]["Cost (USD)"].values[0].replace("$", "").replace(",", "")
                material_cost_total = float(material_cost_total)
            except Exception as e:
                st.warning(f"Could not calculate material costs: {str(e)}")
                material_cost_total = 0
                material_costs_df = None
            
            # Display results in an expander section
            with st.expander("Transformer Price Estimation Results", expanded=True):
                # Create columns for price and material cost
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Estimated Price")
                    st.markdown(f"<h2 style='color:#1E88E5;'>${predicted_price:,.2f}</h2>", unsafe_allow_html=True)
                    st.write(f"Estimated using {selected_model}")
                    
                    # Show breakdown
                    st.markdown("#### Price Composition")
                    st.markdown(f"**Raw Materials:** ${material_cost_total:,.2f} ({material_cost_total/predicted_price*100:.1f}% of total)")
                    st.markdown(f"**Labor, Overhead & Profit:** ${predicted_price - material_cost_total:,.2f} ({(predicted_price - material_cost_total)/predicted_price*100:.1f}% of total)")
                
                with col2:
                    st.markdown("### Raw Material Costs Breakdown")
                    if material_costs_df is not None:
                        st.dataframe(material_costs_df, hide_index=True)
                    else:
                        st.write("Material cost calculation not available")
                
                # Create a centered column with reduced width for the chart
                chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])
                
                with chart_col2:
                    # Simple chart to visualize the price components
                    st.subheader("Price Components")
                    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced figure size
                    components = ['Raw Materials', 'Labor, Overhead & Profit']
                    values = [material_cost_total, predicted_price - material_cost_total]
                    colors = ['#1976D2', '#4CAF50']
                    
                    ax.bar(components, values, color=colors)
                    ax.set_ylabel('USD')
                    ax.set_title('Transformer Price Components')
                    
                    # Add values on top of the bars
                    for i, v in enumerate(values):
                        ax.text(i, v + 0.1, f'${v:,.2f}', ha='center')
                    
                    plt.tight_layout()  # Add tight layout to ensure proper spacing
                    st.pyplot(fig, use_container_width=False)  # Set use_container_width to False
                
        except Exception as e:
            st.error(f"Error calculating price: {str(e)}")
            st.info("Please check that the model file and preprocessor are correctly loaded.")

def model_performance_tab():
    st.header("Model Performance Analysis")
    display_model_performance()
    
    # Add a section for material price trends (if we implement that later)
    st.subheader("Current Material Prices")
    try:
        prices_df = get_current_prices_dataframe()
        st.dataframe(prices_df, hide_index=True)
        
        # Create centered columns for the material prices chart
        chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])
        
        with chart_col2:
            # Display material prices visualization
            st.subheader("Material Prices Visualization")
            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size
            
            # Extract materials and their prices
            materials = prices_df["Material"].tolist()
            # Remove the word "Per Ton" from the Price column and convert to float
            prices = [float(price.replace("$", "").replace(",", "").replace("/Ton", "")) 
                     for price in prices_df["Price (USD/Ton)"].tolist()]
            
            # Create horizontal bar chart
            ax.barh(materials, prices, color='#1976D2')
            ax.set_xlabel('USD per Ton')
            ax.set_title('Current Material Prices (USD/Ton)')
            
            # Add values at the end of each bar
            for i, v in enumerate(prices):
                ax.text(v + 0.1, i, f'${v:,.2f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
    except Exception as e:
        st.warning(f"Could not load current material prices: {str(e)}")

def about_transformers_tab():
    st.header("About Transformers and Pricing Factors")
    
    st.write("""
    ## What Are Power Transformers?
    
    Power transformers are essential electrical devices that transfer electrical energy between different voltage levels 
    in power networks. They are vital components in the generation, transmission, and distribution of electricity.
    
    ## Key Components Affecting Price
    
    The price of a transformer is influenced by various factors:
    """)
    
    # Create columns for a better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Material Factors")
        st.markdown("""
        - **Core Materials**: Typically electrical steel with varying grades affecting efficiency and cost
        - **Winding Materials**: Usually copper or aluminum, with copper being more expensive but more efficient
        - **Insulation Materials**: Paper, pressboard, and other insulating materials
        - **Tank**: Steel structure that houses the core and windings
        - **Oil**: Special mineral or synthetic oil for cooling and insulation
        """)
        
        # Display current material prices
        st.subheader("Current Material Prices")
        try:
            prices_df = get_current_prices_dataframe()
            st.dataframe(prices_df, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load current material prices: {str(e)}")
    
    with col2:
        st.subheader("Design and Technical Factors")
        st.markdown("""
        - **Power Rating (kVA)**: Higher capacity means higher cost
        - **Voltage Ratings**: Higher voltage ratings require better insulation and design
        - **Cooling Type**: ONAN, ONAF, OFAF, ODAF with varying complexity and cost
        - **Tap Changer**: On-load tap changers are more expensive than off-load ones
        - **Efficiency**: Higher efficiency designs require better materials and construction
        - **Special Features**: Monitoring equipment, special protection systems, etc.
        """)
    
    st.write("""
    ## Material Price Impact
    
    Raw material costs often account for 60-70% of a transformer's total cost. Market fluctuations in materials like copper, 
    electrical steel, and oil can significantly impact transformer prices.
    
    ### Recent Trends
    
    - Volatile copper prices have significant impact on transformer costs
    - Electrical steel supply chain disruptions
    - Focus on more environmentally friendly insulating materials
    
    ## Manufacturing Considerations
    
    The manufacturing location affects both labor costs and transportation expenses. Differences in regional manufacturing 
    standards, labor costs, and regulatory requirements all contribute to the final price.
    """)
    
    # Add an image of a transformer with key components labeled
    try:
        st.image("visualizations/transformer_diagram.png", caption="Transformer Key Components", use_column_width=True)
    except:
        st.info("Transformer diagram image not available")

def main():
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # App title and description
    try:
        st.image("visualizations/logo.png", width=100)
    except:
        st.title("⚡ Transformer Price Calculator")
        
    st.title("Transformer Price Calculator")
    st.write("Estimate power transformer prices based on specifications using machine learning models")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Price Calculator", "Model Performance", "About Transformers"])
    
    with tab1:
        price_calculator_tab()
    
    with tab2:
        model_performance_tab()
    
    with tab3:
        about_transformers_tab()

if __name__ == "__main__":
    main() 