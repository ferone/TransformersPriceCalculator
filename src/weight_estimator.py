"""
Weight Estimator Module for Transformer Price Calculator
--------------------------------------------------------
This module provides functions for estimating transformer component weights
based on power ratings and voltage specifications. It uses industry data 
and engineering formulas to produce realistic weight estimations.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union

# MAKSAN transformer data and other industry standards
# Power ratings in kVA
POWER_RATINGS = [25, 50, 100, 160, 250, 400, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 5000, 10000, 20000, 40000, 80000, 120000, 200000]

# Primary voltage levels in kV
PRIMARY_VOLTAGES = [6.3, 10.5, 15.75, 20, 35]

# Secondary voltage levels in kV
SECONDARY_VOLTAGES = [0.4, 0.69, 1.1, 3.3, 6.3]

# Base weights for reference 1000 kVA transformer (kg)
BASE_WEIGHTS = {
    "core": 1200,       # Magnetic core weight
    "copper": 2300,     # Copper winding weight
    "insulation": 600,  # Insulation materials weight
    "tank": 1000,       # Tank and structural components
    "oil": 900,         # Transformer oil weight
}

# Scaling exponents for different materials
SCALING_EXPONENTS = {
    "core": 0.60,       # Further reduced from 0.65 to 0.60 for better large transformer accuracy
    "copper": 0.65,     # Further reduced from 0.70 to 0.65 for better large transformer accuracy
    "insulation": 0.55, # Further reduced from 0.60 to 0.55 for better large transformer accuracy
    "tank": 0.50,       # Further reduced from 0.55 to 0.50 for better large transformer accuracy
    "oil": 0.60,        # Further reduced from 0.70 to 0.60 for better large transformer accuracy
}

# Correction factors for large transformers (MVA range)
# These factors help adjust the weight estimates for very large transformers
LARGE_TRANSFORMER_CORRECTION = {
    "threshold_kva": 5000,  # Apply correction above this kVA rating
    "core": 0.75,           # Improved from 0.85 to 0.75 for better large transformer accuracy 
    "copper": 0.70,         # Improved from 0.80 to 0.70 for better large transformer accuracy
    "insulation": 0.80,     # Improved from 0.90 to 0.80 for better large transformer accuracy
    "tank": 0.75,           # Improved from 0.85 to 0.75 for better large transformer accuracy
    "oil": 0.80,            # Improved from 0.90 to 0.80 for better large transformer accuracy
}

# Voltage adjustment factors - higher voltages require more insulation and larger clearances
VOLTAGE_ADJUSTMENT = {
    "core": 0.05,       # Core weight increases slightly with voltage
    "copper": 0.12,     # Copper weight increases moderately with voltage
    "insulation": 0.25, # Insulation increases significantly with voltage
    "tank": 0.1,        # Tank size increases with voltage for clearances
    "oil": 0.15,        # Oil volume increases with voltage
}

# Phase adjustment factors - single phase vs three phase
PHASE_ADJUSTMENT = {
    "single": {
        "core": 0.7,     # Single phase core is smaller
        "copper": 0.8,   # Single phase uses less copper
        "insulation": 0.85, # Slightly less insulation
        "tank": 0.75,    # Smaller tank
        "oil": 0.7,      # Less oil needed
    },
    "three": {
        "core": 1.0,     # Base case is three phase
        "copper": 1.0,
        "insulation": 1.0,
        "tank": 1.0,
        "oil": 1.0,
    }
}

# Phase format mapping
PHASE_MAPPING = {
    "single": "single",
    "three": "three",
    "Single-phase": "single",
    "Three-phase": "three",
    "single-phase": "single",
    "three-phase": "three",
    "SINGLE-PHASE": "single",
    "THREE-PHASE": "three",
    "Single phase": "single",
    "Three phase": "three"
}

# Special adjustments for ultra-large transformers (>50 MVA)
ULTRA_LARGE_ADJUSTMENTS = {
    "threshold_mva": 50,       # Threshold in MVA where ultra-large adjustments begin
    "max_reduction": 0.35,     # Maximum reduction in weight for largest transformers
    "power_density_target": {  # Target power density by MVA range (in kVA/kg)
        50: 1.5,               # 50 MVA - target around 1.5 kVA/kg
        100: 1.8,              # 100 MVA - target around 1.8 kVA/kg
        200: 2.2               # 200 MVA - target around 2.2 kVA/kg
    }
}

def get_power_ratings() -> List[int]:
    """Return available power ratings for transformers in kVA."""
    return POWER_RATINGS

def get_primary_voltages() -> List[float]:
    """Return available primary voltage levels in kV."""
    return PRIMARY_VOLTAGES

def get_secondary_voltages() -> List[float]:
    """Return available secondary voltage levels in kV."""
    return SECONDARY_VOLTAGES

def normalize_phase(phase_str: str) -> str:
    """
    Normalize phase string to the format used internally ('single' or 'three').
    
    Args:
        phase_str: Phase designation in any format (e.g., 'Single-phase', 'three')
        
    Returns:
        Normalized phase string ('single' or 'three')
        
    Raises:
        ValueError: If the phase string is not recognized
    """
    if not phase_str:
        return "three"  # Default to three-phase
        
    normalized = PHASE_MAPPING.get(phase_str)
    if not normalized:
        # Try to normalize by other means
        phase_lower = phase_str.lower()
        if "single" in phase_lower:
            normalized = "single"
        elif "three" in phase_lower:
            normalized = "three"
        else:
            raise ValueError(f"Unknown phase type: {phase_str}. Must be convertible to 'single' or 'three'.")
            
    return normalized

def estimate_weights_from_power_and_voltage(
    power_rating: float,
    primary_voltage: float,
    secondary_voltage: float,
    phase: str = "three"
) -> Dict[str, float]:
    """
    Estimate transformer component weights based on power rating and voltage levels.
    
    Args:
        power_rating (float): Power rating in kVA
        primary_voltage (float): Primary voltage in kV
        secondary_voltage (float): Secondary voltage in kV
        phase (str): Phase type such as 'single', 'three', 'Single-phase', or 'Three-phase'
        
    Returns:
        Dict[str, float]: Dictionary containing estimated weights for each component with keys
                         such as "core_weight", "copper_weight", etc.
    """
    # Validate inputs
    if power_rating <= 0:
        raise ValueError("Power rating must be positive")
    
    if primary_voltage <= 0 or secondary_voltage <= 0:
        raise ValueError("Voltage values must be positive")
    
    # Normalize the phase
    normalized_phase = normalize_phase(phase)
    
    # Calculate the base reference power ratio
    power_ratio = power_rating / 1000.0  # Relative to 1000 kVA base
    
    # Calculate the voltage complexity factor
    # Higher voltage differential means more complex transformer
    voltage_ratio = (primary_voltage + 0.1) / (secondary_voltage + 0.1)
    voltage_factor = math.log10(voltage_ratio + 1) + 1
    
    # Apply large transformer correction if power rating exceeds threshold
    large_transformer_factor = 1.0
    if power_rating >= LARGE_TRANSFORMER_CORRECTION["threshold_kva"]:
        # Calculate how many times over the threshold we are
        scale_factor = math.log10(power_rating / LARGE_TRANSFORMER_CORRECTION["threshold_kva"] + 1)
        # The correction becomes more significant as the power rating increases
        large_transformer_factor = math.pow(scale_factor, 0.5)  # Changed from 0.3 to 0.5 for stronger effect
    
    # Special compensation for ultra-large transformers (>50 MVA)
    ultra_large_compensation = 1.0
    if power_rating > ULTRA_LARGE_ADJUSTMENTS["threshold_mva"] * 1000:  # Convert MVA to kVA
        # Calculate MVA for easier comparisons
        power_mva = power_rating / 1000
        
        # Additional scaling factor that grows logarithmically with size
        ultra_large_factor = math.log10(power_mva / ULTRA_LARGE_ADJUSTMENTS["threshold_mva"] + 1)
        
        # Calculate maximum reduction percentage based on size
        max_reduction_percentage = ULTRA_LARGE_ADJUSTMENTS["max_reduction"] * ultra_large_factor
        
        # Cap the reduction to avoid unrealistic values
        if max_reduction_percentage > ULTRA_LARGE_ADJUSTMENTS["max_reduction"]:
            max_reduction_percentage = ULTRA_LARGE_ADJUSTMENTS["max_reduction"]
            
        # Apply the reduction
        ultra_large_compensation = 1.0 - max_reduction_percentage
    
    # Calculate estimated weights
    weights = {}
    total_weight = 0
    
    # Component name mapping to ensure consistent naming
    component_output_keys = {
        "core": "core_weight",
        "copper": "copper_weight",
        "insulation": "insulation_weight",
        "tank": "tank_weight",
        "oil": "oil_weight"
    }
    
    # Component-specific adjustment factors for ultra-large transformers
    # Different components scale differently at ultra-large sizes
    ultra_large_component_factors = {
        "core": 1.0,       # Core doesn't benefit as much from economies of scale
        "copper": 0.9,     # Copper windings benefit more from optimal design
        "insulation": 0.85, # Insulation benefits from advanced materials
        "tank": 0.95,      # Tank doesn't benefit as much from economies of scale
        "oil": 0.9         # Oil benefits from more efficient cooling designs
    }
    
    for component in BASE_WEIGHTS:
        # Apply power scaling with appropriate exponent
        weight = BASE_WEIGHTS[component] * (power_ratio ** SCALING_EXPONENTS[component])
        
        # Apply voltage adjustment - higher voltages increase weight
        voltage_adjustment = 1.0 + (VOLTAGE_ADJUSTMENT[component] * (voltage_factor - 1))
        weight *= voltage_adjustment
        
        # Apply phase adjustment
        weight *= PHASE_ADJUSTMENT[normalized_phase][component]
        
        # Apply large transformer correction if applicable
        if power_rating >= LARGE_TRANSFORMER_CORRECTION["threshold_kva"]:
            # Apply component-specific large transformer correction
            correction = LARGE_TRANSFORMER_CORRECTION[component]
            # The correction increases with transformer size
            correction_factor = 1.0 - ((1.0 - correction) * large_transformer_factor)
            weight *= correction_factor
        
        # Apply ultra-large transformer compensation (>50 MVA)
        if power_rating > ULTRA_LARGE_ADJUSTMENTS["threshold_mva"] * 1000:
            # Apply component-specific adjustment factor
            weight *= ultra_large_compensation * ultra_large_component_factors[component]
        
        # Add some realistic variation (±5%)
        weight *= np.random.uniform(0.95, 1.05)
        
        # Round to nearest kilogram
        weight = round(weight)
        
        # Use the output key format expected by the app
        output_key = component_output_keys[component]
        weights[output_key] = weight
        total_weight += weight
    
    # Add total weight
    weights["total_weight"] = total_weight
    
    # Power density check for ultra-large transformers
    # If power density is below target, adjust weights to achieve target
    if power_rating > ULTRA_LARGE_ADJUSTMENTS["threshold_mva"] * 1000:
        power_mva = power_rating / 1000
        current_power_density = power_rating / total_weight
        
        # Determine target power density through interpolation
        target_density = None
        for target_mva in sorted(ULTRA_LARGE_ADJUSTMENTS["power_density_target"].keys()):
            if power_mva <= target_mva:
                if target_density is None:
                    # First target found, use it directly
                    target_density = ULTRA_LARGE_ADJUSTMENTS["power_density_target"][target_mva]
                break
            target_density = ULTRA_LARGE_ADJUSTMENTS["power_density_target"][target_mva]
        
        # If we're beyond the largest defined MVA, use the highest defined target
        if target_density is None:
            largest_mva = max(ULTRA_LARGE_ADJUSTMENTS["power_density_target"].keys())
            target_density = ULTRA_LARGE_ADJUSTMENTS["power_density_target"][largest_mva]
        
        # If current density is too low, adjust weights
        if current_power_density < target_density * 0.9:  # Allow 10% below target
            # Calculate adjustment factor to reach target
            adjustment_factor = current_power_density / target_density
            
            # Apply adjustment to each component
            for component, output_key in component_output_keys.items():
                weights[output_key] = round(weights[output_key] * adjustment_factor)
            
            # Recalculate total weight
            total_weight = sum(weights[k] for k in component_output_keys.values())
            weights["total_weight"] = total_weight
    
    return weights

def get_recommended_weight_ranges(power_rating: float, phase: str = "three") -> Dict[str, Tuple[float, float]]:
    """
    Get recommended weight ranges for components based on power rating.
    
    Args:
        power_rating (float): Power rating in kVA
        phase (str): Phase type such as 'single', 'three', 'Single-phase', or 'Three-phase'
        
    Returns:
        Dict[str, Tuple[float, float]]: Dictionary with min/max weight ranges for each component
    """
    # Normalize the phase
    normalized_phase = normalize_phase(phase)
    
    # First estimate a reference point
    reference = estimate_weights_from_power_and_voltage(
        power_rating=power_rating,
        primary_voltage=10.5,  # Common primary voltage
        secondary_voltage=0.4,  # Common secondary voltage
        phase=normalized_phase
    )
    
    # Component name mapping to match estimate function
    component_mapping = {
        "core": "core_weight",
        "copper": "copper_weight",
        "insulation": "insulation_weight",
        "tank": "tank_weight",
        "oil": "oil_weight"
    }
    
    # Create ranges with typical variations
    ranges = {}
    for component, output_key in component_mapping.items():
        min_weight = reference[output_key] * 0.85  # 15% below typical
        max_weight = reference[output_key] * 1.15  # 15% above typical
        ranges[output_key] = (round(min_weight), round(max_weight))
    
    return ranges

def calculate_power_density(total_weight: float, power_rating: float) -> float:
    """
    Calculate power density in kVA/kg.
    
    Args:
        total_weight (float): Total transformer weight in kg
        power_rating (float): Power rating in kVA
        
    Returns:
        float: Power density in kVA/kg
    """
    if total_weight <= 0:
        return 0
    return power_rating / total_weight

def calculate_weight_distribution(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate weight distribution percentages.
    
    Args:
        weights (Dict[str, float]): Dictionary of component weights
        
    Returns:
        Dict[str, float]: Dictionary of weight distribution percentages
    """
    # Extract components excluding total weight
    component_weights = {k: v for k, v in weights.items() if k != "total_weight"}
    
    total = weights.get("total_weight", sum(component_weights.values()))
    
    if total <= 0:
        return {k: 0 for k in component_weights}
    
    return {k: (w / total * 100) for k, w in component_weights.items()}

if __name__ == "__main__":
    # Test the weight estimation
    test_power_ratings = [100, 500, 1000, 5000, 10000, 50000, 90000, 200000]
    
    print("Example Weight Estimations:")
    print("--------------------------")
    
    # Test different phase formats
    test_phases = ["single", "three", "Single-phase", "Three-phase"]
    
    for phase in test_phases:
        print(f"\nTesting phase format: '{phase}'")
        weights = estimate_weights_from_power_and_voltage(
            power_rating=1000,
            primary_voltage=10.5,
            secondary_voltage=0.4,
            phase=phase
        )
        print(f"Total Weight: {weights['total_weight']} kg")
    
    for power in test_power_ratings:
        weights = estimate_weights_from_power_and_voltage(
            power_rating=power,
            primary_voltage=10.5,
            secondary_voltage=0.4
        )
        
        print(f"\n{power} kVA Transformer ({power/1000:.1f} MVA):")
        # Display results with new key names
        component_display_names = {
            "core_weight": "Core",
            "copper_weight": "Copper",
            "insulation_weight": "Insulation",
            "tank_weight": "Tank",
            "oil_weight": "Oil"
        }
        
        for key, display_name in component_display_names.items():
            weight = weights[key]
            print(f"  {display_name}: {weight} kg ({weight/weights['total_weight']*100:.1f}%)")
        
        print(f"  Total Weight: {weights['total_weight']} kg ({weights['total_weight']/1000:.2f} tons)")
        print(f"  Power Density: {calculate_power_density(weights['total_weight'], power):.3f} kVA/kg") 