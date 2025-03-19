from src.weight_estimator import estimate_weights_from_power_and_voltage

# Test for various transformer sizes
test_cases = [
    (1000, 10.5, 0.4),    # 1 MVA transformer
    (5000, 20.0, 0.4),    # 5 MVA transformer
    (10000, 33.0, 11.0),  # 10 MVA transformer
    (50000, 110.0, 11.0), # 50 MVA transformer
    (90000, 115.0, 13.8), # 90 MVA transformer (should be ~50 tons)
    (200000, 230.0, 13.8) # 200 MVA transformer (should be ~90 tons)
]

print("Testing weight estimator for different transformer sizes:")
print("--------------------------------------------------------")

for power, primary_v, secondary_v in test_cases:
    weights = estimate_weights_from_power_and_voltage(power, primary_v, secondary_v, 'three')
    total_weight = weights["total_weight"]
    print(f"{power/1000:.1f} MVA transformer at {primary_v}/{secondary_v} kV: {total_weight/1000:.2f} tons")
    
    # Print component breakdown for the 90MVA and 200MVA transformers
    if power in [90000, 200000]:
        print(f"  Component breakdown:")
        component_display_names = {
            "core_weight": "Core",
            "copper_weight": "Copper",
            "insulation_weight": "Insulation",
            "tank_weight": "Tank",
            "oil_weight": "Oil"
        }
        for key, display_name in component_display_names.items():
            weight = weights[key]
            print(f"    {display_name}: {weight/1000:.2f} tons ({weight/total_weight*100:.1f}%)")
        print(f"  Power Density: {power/total_weight:.2f} kVA/kg")
    print() 