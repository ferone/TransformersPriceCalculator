import streamlit as st
import math

# Page configuration must be the first streamlit command
st.set_page_config(
    page_title="Transformer Price Calculator",
    page_icon="visualizations/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    from src.weight_estimator import estimate_weights_from_power_and_voltage
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
    
    try:
        from src.weight_estimator import estimate_weights_from_power_and_voltage
    except ImportError:
        def estimate_weights_from_power_and_voltage(power_kva, primary_voltage_kv, secondary_voltage_kv=None, phase="Three-phase"):
            """Fallback function for weight estimation"""
            # Simple estimation formula
            total_weight = power_kva * 3.5  # 3.5 kg per kVA is a rough industry estimate
            return {
                "core_weight": round(total_weight * 0.25, 1),     # 25% core
                "copper_weight": round(total_weight * 0.20, 1),   # 20% copper
                "insulation_weight": round(total_weight * 0.10, 1), # 10% insulation
                "tank_weight": round(total_weight * 0.15, 1),     # 15% tank
                "oil_weight": round(total_weight * 0.30, 1),      # 30% oil
                "total_weight": round(total_weight, 1)
            }

# Add CSS to improve the appearance and layout of the app
st.markdown("""
<style>
    .ikutu-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0;
        padding-bottom: 0;
        text-align: left;
        letter-spacing: 0.05em;
        color: #2c3e50;
    }
    
    .ikutu-slogan {
        color: #3498db;
        margin-top: 0;
        text-transform: uppercase;
        font-size: 1rem;
        font-weight: 500;
        text-align: left;
    }
    
    .content-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
        st.image(image, use_container_width=True)

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

    # Display current material prices in a collapsible section (collapsed by default)
    with st.expander("Current Material Prices (USD/Ton)", expanded=False):
        try:
            prices_df = get_current_prices_dataframe()
            st.dataframe(prices_df, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load current material prices: {str(e)}")
            st.write("Using default material prices")
    
    # Session state initialization for weight values
    if 'power_rating' not in st.session_state:
        st.session_state.power_rating = 1000.0
    if 'primary_voltage' not in st.session_state:
        st.session_state.primary_voltage = 33.0
    if 'secondary_voltage' not in st.session_state:
        st.session_state.secondary_voltage = 11.0
    if 'phase' not in st.session_state:
        st.session_state.phase = "Three-phase"
    if 'weights_estimated' not in st.session_state:
        st.session_state.weights_estimated = False
        
        # Initialize with estimated weights
        weights = estimate_weights_from_power_and_voltage(
            st.session_state.power_rating,
            st.session_state.primary_voltage,
            st.session_state.secondary_voltage,
            st.session_state.phase
        )
        
        st.session_state.core_weight = float(weights["core_weight"])
        st.session_state.copper_weight = float(weights["copper_weight"])
        st.session_state.insulation_weight = float(weights["insulation_weight"])
        st.session_state.tank_weight = float(weights["tank_weight"])
        st.session_state.oil_weight = float(weights["oil_weight"])
        st.session_state.total_weight = float(weights["total_weight"])
        st.session_state.weights_estimated = True
    
    # Add weight calculator section outside the form
    st.subheader("Quick Weight Estimator")
    
    # Create a layout with 3 columns: 1 for inputs and 2 for results
    inputs_col, results_col = st.columns([1, 2])
    
    # All inputs in the first column
    with inputs_col:
        est_power_rating = st.number_input(
            "Power Rating (kVA)", 
            min_value=10.0, 
            max_value=100000.0, 
            value=1000.0,
            key="est_power_rating_input"
        )
        
        est_phase = st.selectbox(
            "Phase",
            ["Single-phase", "Three-phase"],
            index=1 if st.session_state.phase == "Three-phase" else 0,
            key="est_phase_input"
        )
        
        est_primary_voltage = st.number_input(
            "Primary Voltage (kV)", 
            min_value=0.1, 
            max_value=1000.0, 
            value=33.0,
            key="est_primary_voltage_input"
        )
        
        est_secondary_voltage = st.number_input(
            "Secondary Voltage (kV)", 
            min_value=0.1, 
            max_value=1000.0, 
            value=11.0,
            key="est_secondary_voltage_input"
        )
        
        # Add some spacing before the button
        st.write("")
        
        # Estimate weights button at the bottom of input column
        estimate_button = st.button("Estimate Weights", key="estimate_weights_button", use_container_width=True)
    
    # Results will be displayed in the wider right column
    with results_col:
        if estimate_button:
            try:
                weights = estimate_weights_from_power_and_voltage(
                    est_power_rating,
                    est_primary_voltage,
                    est_secondary_voltage,
                    est_phase
                )
                
                # Validate that the weights are reasonable
                if max(weights.values()) > 1000000:  # Arbitrary upper bound
                    st.warning("Calculated weights seem unusually large. Please double-check your inputs.")
                    # Still update but provide warning
                
                # Update session state
                st.session_state.power_rating = est_power_rating
                st.session_state.primary_voltage = est_primary_voltage
                st.session_state.phase = est_phase
                st.session_state.core_weight = float(weights["core_weight"])
                st.session_state.copper_weight = float(weights["copper_weight"])
                st.session_state.insulation_weight = float(weights["insulation_weight"])
                st.session_state.tank_weight = float(weights["tank_weight"])
                st.session_state.oil_weight = float(weights["oil_weight"])
                st.session_state.total_weight = float(weights["total_weight"])
                
                # Show success message
                st.success(f"Weights estimated based on {est_power_rating} kVA at {est_primary_voltage} kV!")
                
                # Display the weight distribution chart
                display_weight_distribution(
                    st.session_state.core_weight,
                    st.session_state.copper_weight,
                    st.session_state.insulation_weight,
                    st.session_state.tank_weight,
                    st.session_state.oil_weight,
                    est_power_rating
                )
            except Exception as e:
                st.error(f"Error estimating weights: {str(e)}")
                st.info("Please try with different values or enter weights manually.")
    
    # Create form for transformer specifications
    with st.form("transformer_specs_form"):
        st.subheader("Transformer Specifications")
        
        # Create columns for a more compact layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            power_rating = st.number_input(
                "Power Rating (kVA)", 
                min_value=10.0, 
                max_value=100000.0, 
                value=float(st.session_state.power_rating),
                key="power_rating_input"
            )
            st.session_state.power_rating = power_rating
            
            primary_voltage = st.number_input(
                "Primary Voltage (kV)", 
                min_value=0.1, 
                max_value=1000.0, 
                value=st.session_state.primary_voltage,
                key="primary_voltage_input"
            )
            st.session_state.primary_voltage = primary_voltage
            
            secondary_voltage = st.number_input(
                "Secondary Voltage (kV)", 
                min_value=0.1, 
                max_value=1000.0, 
                value=st.session_state.secondary_voltage,
                key="secondary_voltage_input"
            )
            st.session_state.secondary_voltage = secondary_voltage
            
        with col2:
            core_weight = st.number_input(
                "Core Weight (kg)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.core_weight),
                key="core_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.core_weight = core_weight
            
            copper_weight = st.number_input(
                "Copper Weight (kg)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.copper_weight),
                key="copper_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.copper_weight = copper_weight
            
            insulation_weight = st.number_input(
                "Insulation Weight (kg)",
                min_value=0.0,
                max_value=50000.0,
                value=float(st.session_state.insulation_weight),
                key="insulation_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.insulation_weight = insulation_weight
            
        with col3:
            tank_weight = st.number_input(
                "Tank Weight (kg)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.tank_weight),
                key="tank_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.tank_weight = tank_weight
            
            oil_weight = st.number_input(
                "Oil Weight (kg)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.oil_weight),
                key="oil_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.oil_weight = oil_weight
            
            total_weight = st.number_input(
                "Total Weight (kg)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.total_weight),
                key="total_weight_input"  # Changed key to avoid conflict
            )
            # Update session state after widget interaction
            st.session_state.total_weight = total_weight
        
        # Help text for weight estimation
        st.info("⚖️ Use the 'Quick Weight Estimator' section above to automatically calculate weights based on power and voltage.")
        
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
                index=1 if st.session_state.phase == "Three-phase" else 0,
                key="phase_input"
            )
            st.session_state.phase = phase
            
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
            
        # Make the submit button more prominent
        st.markdown("### Calculate Price")
        submit_button = st.form_submit_button(
            "Calculate Transformer Price", 
            use_container_width=True, 
            type="primary"  # Make it a primary button for emphasis
        )

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
                    
                    # Apply scale factor for large transformers (economies of scale)
                    labor_overhead_profit_pct = (predicted_price - material_cost_total) / predicted_price * 100
                    
                    # For large transformers, adjust the labor, overhead & profit percentage
                    # Transformers above 5000 kVA are considered large
                    if power_rating > 5000:
                        # Calculate how many times over the large transformer threshold
                        scale_factor = math.log10(power_rating / 5000 + 1)
                        
                        # Calculate target percentage (reduce from current to 30-40% range for largest transformers)
                        # Cap the reduction to ensure min 30% for labor, overhead & profit
                        min_lop_percentage = 30  # minimum percentage for labor, overhead & profit
                        target_percentage = max(min_lop_percentage, labor_overhead_profit_pct * (1 - scale_factor * 0.2))
                        
                        # Adjust the predicted price to achieve the target LOP percentage
                        adjusted_lop_amount = (material_cost_total * target_percentage) / (100 - target_percentage)
                        adjusted_price = material_cost_total + adjusted_lop_amount
                        
                        st.markdown(f"**Raw Materials:** ${material_cost_total:,.2f} ({material_cost_total/adjusted_price*100:.1f}% of total)")
                        st.markdown(f"**Labor, Overhead & Profit:** ${adjusted_lop_amount:,.2f} ({target_percentage:.1f}% of total)")
                        st.markdown(f"**Adjusted Total Price:** ${adjusted_price:,.2f}")
                        st.info(f"Price adjusted for large transformer economies of scale (original estimate: ${predicted_price:,.2f})")
                        
                        # Update the values for the chart
                        predicted_price = adjusted_price
                        values = [material_cost_total, adjusted_lop_amount]
                    else:
                        st.markdown(f"**Raw Materials:** ${material_cost_total:,.2f} ({material_cost_total/predicted_price*100:.1f}% of total)")
                        st.markdown(f"**Labor, Overhead & Profit:** ${predicted_price - material_cost_total:,.2f} ({(predicted_price - material_cost_total)/predicted_price*100:.1f}% of total)")
                        values = [material_cost_total, predicted_price - material_cost_total]
                
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
                    values = values  # Use the values set above
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

def display_weight_distribution(core_weight, copper_weight, insulation_weight, tank_weight, oil_weight, power_rating):
    """Helper function to display weight distribution visualization"""
    st.subheader("Weight Distribution")
    
    # Remove the column layout to allow the chart to use more space
    # Calculate current weights
    current_weights = {
        "Core": core_weight,
        "Copper": copper_weight,
        "Insulation": insulation_weight,
        "Tank": tank_weight,
        "Oil": oil_weight
    }
    
    # Calculate total and percentages
    current_total = sum(current_weights.values())
    if current_total > 0:  # Avoid division by zero
        # Create pie chart with larger figure size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define custom colors for better visualization
        colors = ['#1976D2', '#FFA000', '#4CAF50', '#9C27B0', '#F44336']
        
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            current_weights.values(), 
            labels=current_weights.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 12}  # Increase font size for labels
        )
        
        # Equal aspect ratio ensures the pie chart is circular
        ax.axis('equal')
        
        # Customize the appearance with larger text sizes
        plt.setp(autotexts, size=12, weight="bold")  # Increased from 9 to 12
        plt.setp(texts, size=14)  # Increased from 10 to 14
        
        ax.set_title('Transformer Weight Distribution', fontsize=16)  # Increased title size
        
        # Add a legend with weight values and larger font
        legend_labels = [f"{name}: {weight:,.1f} kg ({weight/current_total*100:.1f}%)" 
                        for name, weight in current_weights.items()]
        ax.legend(wedges, legend_labels, title="Components", 
                title_fontsize=14,  # Increase title font size
                fontsize=12,  # Increase legend text size
                loc="center left", 
                bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Adjust layout to accommodate the larger chart
        plt.tight_layout(pad=2.0)
        
        # Use full container width for better visibility
        st.pyplot(fig, use_container_width=True)
        
        # Add informational text with larger font
        st.markdown(f"**Total Weight: {current_total:,.1f} kg | Power Density: {power_rating/current_total:.2f} kVA/kg**")
    else:
        st.info("Please enter non-zero weight values to see the distribution chart.")

def model_performance_tab():
    st.header("Model Performance")
    
    try:
        # Load model performance data
        performance_df = pd.read_csv("models/model_performance_summary.csv")
        
        # Display model performance table
        st.dataframe(performance_df)
        
        # Create bar chart of model performance metrics
        st.subheader("Model Comparison")
        
        # RMSE Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(performance_df['Model'], performance_df['RMSE'])
        ax.set_title('Root Mean Square Error (RMSE) by Model')
        ax.set_ylabel('RMSE (lower is better)')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        # R² Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(performance_df['Model'], performance_df['R²'])
        ax.set_title('R² Score by Model')
        ax.set_ylabel('R² (higher is better)')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading model performance data: {str(e)}")
        st.info("Train models first to generate performance metrics")

def training_data_tab():
    st.header("Training Data")
    st.write("This tab displays the synthetic data used to train the transformer price prediction models.")
    
    # Try to load the training data
    try:
        # First try the direct path
        if os.path.exists("data/transformer_data.csv"):
            data_path = "data/transformer_data.csv"
        # Then try the relative path
        elif os.path.exists("../data/transformer_data.csv"):
            data_path = "../data/transformer_data.csv"
        # If neither exists, generate new data
        else:
            st.info("Training data file not found. Generating new synthetic data...")
            
            # Import the data generation function
            try:
                # Try to import weight_estimator first since data_generation depends on it
                try:
                    from src.weight_estimator import estimate_weights_from_power_and_voltage
                except ImportError:
                    # Try relative import
                    sys.path.append(os.path.abspath("."))
                    from weight_estimator import estimate_weights_from_power_and_voltage
                
                # Now try to import data_generation
                try:
                    from src.data_generation import generate_synthetic_data
                except ImportError:
                    # Try relative import if the first one fails
                    from data_generation import generate_synthetic_data
            except ImportError as ie:
                st.error(f"Could not import required modules: {str(ie)}")
                st.info("Make sure the src directory contains weight_estimator.py and data_generation.py files.")
                return
            
            # Generate sample data
            df = generate_synthetic_data(n_samples=500)
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Save the data for future use
            df.to_csv("data/transformer_data.csv", index=False)
            st.success("Generated new training data and saved to 'data/transformer_data.csv'")
            
            data_path = "data/transformer_data.csv"
        
        # Read the data
        df = pd.read_csv(data_path)
        
        # Add information about the dataset
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Samples", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns) - 1)  # Exclude price column
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        # Display correlation with price
        st.subheader("Feature Correlations with Price")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create a new DataFrame for correlation analysis
        # Keep only numerical columns initially
        df_corr = df[numerical_cols].copy()
        
        # Add dummy variables for categorical columns
        for col in categorical_cols:
            # Skip if column is all NaN
            if df[col].isna().all():
                continue
                
            # Create dummy variables and add to correlation DataFrame
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df_corr = pd.concat([df_corr, dummies], axis=1)
        
        # Calculate correlations with price
        if 'price' in df_corr.columns:
            correlations = df_corr.corr()['price'].sort_values(ascending=False)
            
            # Filter out the price correlation with itself
            correlations = correlations.drop('price')
            
            # Display top 15 positive correlations and top 15 negative correlations
            st.write("Top correlations with price:")
            
            # Create two columns for positive and negative correlations
            pos_col, neg_col = st.columns(2)
            
            with pos_col:
                st.write("Positive correlations:")
                st.dataframe(correlations.head(15))
            
            with neg_col:
                st.write("Negative correlations:")
                st.dataframe(correlations.tail(15))
            
            # Create bar chart of correlations (top 10 positive and top 10 negative)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get top 10 positive and negative correlations
            top_positive = correlations.head(10)
            top_negative = correlations.tail(10).iloc[::-1]  # Reverse order for display
            
            # Combine and sort for better visualization
            top_correlations = pd.concat([top_positive, top_negative])
            
            # Plot
            colors = ['green' if x > 0 else 'red' for x in top_correlations]
            top_correlations.plot(kind='barh', ax=ax, color=colors)
            ax.set_title('Top Feature Correlations with Price')
            ax.set_xlabel('Correlation Coefficient')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            st.pyplot(fig)
        else:
            st.warning("Price column not found in the dataset. Cannot calculate correlations.")
        
        # Allow user to explore the raw data
        st.subheader("Raw Training Data")
        st.dataframe(df)
        
        # Add data download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Training Data as CSV",
            data=csv,
            file_name="transformer_training_data.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error loading or processing training data: {str(e)}")
        st.info("Unable to display training data. Check if the data directory exists and contains transformer_data.csv file.")
        
        # Add option to generate new data if error occurs
        if st.button("Generate New Training Data"):
            try:
                # Try to import weight_estimator first since data_generation depends on it
                try:
                    from src.weight_estimator import estimate_weights_from_power_and_voltage
                except ImportError:
                    # Try relative import
                    sys.path.append(os.path.abspath("."))
                    from weight_estimator import estimate_weights_from_power_and_voltage
                
                # Now try to import data_generation
                try:
                    from src.data_generation import generate_synthetic_data
                except ImportError:
                    # Try relative import if the first one fails
                    from data_generation import generate_synthetic_data
                
                # Generate sample data
                df = generate_synthetic_data(n_samples=500)
                
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                # Save the data for future use
                df.to_csv("data/transformer_data.csv", index=False)
                st.success("Generated new training data and saved to 'data/transformer_data.csv'")
                st.info("Please refresh the page to view the data.")
            except Exception as gen_error:
                st.error(f"Error generating new data: {str(gen_error)}")
                st.code(traceback.format_exc(), language="python")

def machine_learning_tab():
    st.header("Understanding Machine Learning Models")
    
    st.write("""
    ## How Machine Learning Models Predict Transformer Prices
    
    This section explains how each machine learning model works to predict transformer prices and how different 
    parameters influence the final price prediction. Understanding these relationships can help in making informed 
    decisions about transformer specifications and cost optimization.
    """)
    
    # Get available models
    models = get_available_models()
    
    if not models:
        st.warning("No trained models found. Please make sure the models directory exists and contains trained model files.")
        return
    
    # Create expandable sections for each model type
    with st.expander("Gradient Boosting Regressor", expanded=True):
        st.write("""
        ### Gradient Boosting Regressor
        
        Gradient Boosting builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous ones.
        
        #### How It Works
        - Creates multiple decision trees that work together to make a prediction
        - Each tree focuses on correcting the mistakes of previous trees
        - Combines many "weak learners" into a strong predictive model
        - Very effective for complex non-linear relationships in data
        
        #### Key Parameter Influences
        - **Power Rating**: Highly significant - larger transformers typically cost more in a non-linear relationship
        - **Material Weights**: Direct correlation with cost - copper weight has particularly high importance
        - **Voltage Ratings**: Higher voltage ratings require more sophisticated insulation and design
        - **Cooling Type**: More advanced cooling systems (OFAF, ODAF) increase price substantially
        - **Manufacturing Location**: Labor costs vary significantly by region
        
        #### Advantages
        - Captures complex interactions between features
        - Handles both categorical and numerical data effectively
        - Usually achieves the highest accuracy among all models
        """)
        
        # Show feature importance if available
        try:
            display_feature_importance("gradient_boosting")
        except:
            st.info("Feature importance visualization not available for this model")
    
    with st.expander("Random Forest Regressor"):
        st.write("""
        ### Random Forest Regressor
        
        Random Forest creates multiple decision trees using random subsets of data and features, then averages their predictions.
        
        #### How It Works
        - Builds many decision trees in parallel
        - Each tree uses a random subset of features and training data
        - Combines predictions by averaging (for regression) or voting (for classification)
        - Reduces overfitting through randomization
        
        #### Key Parameter Influences
        - **Raw Materials**: Weights of copper, steel, and oil typically account for 60-70% of total cost
        - **Power Rating and Voltage**: Define the transformer's basic capacity requirements
        - **Efficiency**: Higher efficiency requires better materials, affecting cost exponentially as efficiency approaches 100%
        - **Technical Features**: Special requirements like on-load tap changers add fixed cost components
        
        #### Advantages
        - Less prone to overfitting than single decision trees
        - Good at handling outliers in the data
        - Provides reliable feature importance measures
        """)
        
        # Show feature importance if available
        try:
            display_feature_importance("random_forest")
        except:
            st.info("Feature importance visualization not available for this model")
    
    with st.expander("Linear Regression Models (Linear, Ridge, Lasso, Elastic Net)"):
        st.write("""
        ### Linear Regression Models
        
        Linear models establish direct linear relationships between input features and the price.
        
        #### How They Work
        - **Linear Regression**: Simple model that finds the best fit line between features and price
        - **Ridge Regression**: Adds a penalty on the size of coefficients to prevent overfitting
        - **Lasso Regression**: Similar to Ridge but can reduce coefficients to zero, performing feature selection
        - **Elastic Net**: Combines Ridge and Lasso approaches
        
        #### Key Parameter Influences
        - These models find direct linear relationships between features and price
        - Material weights have a nearly 1:1 relationship with cost once material prices are factored in
        - Power rating shows a strong linear correlation with price
        - Categorical features like cooling type and tap changer type add fixed amounts to the base price
        
        #### Limitations
        - Cannot capture complex non-linear relationships without feature engineering
        - May underestimate prices for high-end transformers with special requirements
        - May not accurately model interaction effects between features
        """)
        
        # Create a diagram to illustrate linear relationships
        st.write("#### Linear Relationship Example")
        col1, col2 = st.columns([1, 2])
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Simulate power rating vs price relationship
            power_ratings = np.linspace(100, 10000, 100)
            base_price = 5000
            price_per_kva = 25
            
            # Create different lines for different models
            linear_prices = base_price + price_per_kva * power_ratings
            nonlinear_prices = base_price + power_ratings * (20 + 0.001 * power_ratings)
            
            ax.plot(power_ratings, linear_prices, label='Linear Model', color='blue')
            ax.plot(power_ratings, nonlinear_prices, label='Gradient Boosting', color='red')
            
            ax.set_xlabel('Power Rating (kVA)')
            ax.set_ylabel('Price (USD)')
            ax.set_title('Power Rating vs. Price Relationship by Model Type')
            ax.legend()
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=False)
        
        with col1:
            st.write("""
            This chart shows how different models interpret the relationship between power rating and price:
            
            - **Linear models** assume a constant price increase per kVA
            - **Tree-based models** (like Gradient Boosting) can capture non-linear price increases that occur at higher power ratings
            
            In reality, larger transformers often have economies of scale in some aspects but require exponentially more complex engineering in others.
            """)
    
    # General insights section
    st.subheader("Key Insights Across All Models")
    
    st.write("""
    ### Most Influential Parameters
    
    Across all models, these factors consistently show the strongest influence on transformer prices:
    
    1. **Material Weights and Costs**
       - Copper weight is typically the most expensive material component
       - Core material (electrical steel) quality affects both price and efficiency
       - Oil quantity increases with size but also depends on cooling requirements
    
    2. **Power Rating**
       - Fundamental determinant of transformer size and capacity
       - Higher power ratings require more materials and more complex designs
    
    3. **Voltage Ratings**
       - Higher voltage transformers require better insulation and clearances
       - Affects safety requirements and testing procedures
    
    4. **Manufacturing Location**
       - Labor costs vary significantly by country/region
       - Different regions may have different regulatory requirements
    
    5. **Cooling Type**
       - More advanced cooling systems add complexity and cost
       - Required for higher power density designs
    
    ### How to Use This Knowledge
    
    Understanding these relationships can help in:
    - Optimizing transformer specifications for cost efficiency
    - Making informed trade-offs between performance and cost
    - Predicting how market changes (like material price fluctuations) will affect transformer prices
    - Planning budgets for transformer procurement projects
    """)

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
    ## Material Weight Estimation
    
    The weight of transformer components can be estimated based on its power rating and voltage specifications. 
    Higher power ratings require more materials, and higher voltages require improved insulation and larger clearances.
    
    ### Weight Distribution in Typical Transformers
    
    - **Core**: Usually 15-25% of total weight, higher percentage in smaller transformers
    - **Windings**: Typically 15-25% of total weight, varies with design and material (copper or aluminum)
    - **Insulation**: Around 5-15% of total weight, higher percentage in high-voltage transformers 
    - **Tank**: About 15-25% of total weight
    - **Oil**: About 25-35% of total weight in oil-filled transformers
    
    Our calculator uses data from MAKSAN and industry standard formulas to estimate component weights based on your specifications.
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
        st.image("visualizations/transformer_diagram.png", caption="Transformer Key Components", use_container_width=True)
    except:
        st.info("Transformer diagram image not available")

# Add a function to load and encode the image as base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            import base64
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return ""

def main():
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Create a layout with tighter columns for logo and title
    logo_col, title_col, spacer_col = st.columns([1, 3, 2])
    
    # Logo in the first column
    with logo_col:
        try:
            st.image("visualizations/logo.png", width=100)
        except Exception as e:
            st.write("Logo not found")
    
    # Title and slogan in the second column, with CSS to align it closer to the logo
    with title_col:
        st.markdown("""
        <div class="content-container" style="margin-left: -20px; margin-top: 10px;">
            <h1 class="ikutu-title">IKUTU</h1>
            <p class="ikutu-slogan">SUSTAINABLE ENERGY SOLUTIONS</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.title("Transformer Price Calculator")
    st.write("Estimate power transformer prices based on specifications using machine learning models")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Calculator", "Model Performance", "Training Data", "Machine Learning", "About Transformers"])
    
    with tab1:
        price_calculator_tab()
    
    with tab2:
        model_performance_tab()
    
    with tab3:
        training_data_tab()
    
    with tab4:
        machine_learning_tab()
    
    with tab5:
        about_transformers_tab()

if __name__ == "__main__":
    main() 