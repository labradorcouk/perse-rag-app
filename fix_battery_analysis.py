import pandas as pd
import numpy as np

# Assuming df1 contains the PHEV data
# Let's analyze the battery capacity data

# First, let's check what battery-related columns we have
battery_columns = [col for col in df1.columns if 'Battery' in col]
print("Battery-related columns:", battery_columns)

# Check for any charging-related columns
charging_columns = [col for col in df1.columns if 'Charge' in col or 'Charging' in col]
print("Charging-related columns:", charging_columns)

# Since we don't have direct charge time, let's analyze battery capacity
# Find the vehicle with the highest full battery capacity
if 'Battery_Capacity_Full' in df1.columns:
    # Remove any null values
    df1_clean = df1.dropna(subset=['Battery_Capacity_Full'])
    
    # Find the vehicle with highest battery capacity
    max_battery_idx = df1_clean['Battery_Capacity_Full'].idxmax()
    max_battery_vehicle = df1_clean.loc[max_battery_idx]
    
    print("\n=== Vehicle with Highest Battery Capacity ===")
    print(f"Vehicle ID: {max_battery_vehicle['Vehicle_ID']}")
    print(f"Make: {max_battery_vehicle['Vehicle_Make']}")
    print(f"Model: {max_battery_vehicle['Vehicle_Model']}")
    print(f"Battery Capacity (Full): {max_battery_vehicle['Battery_Capacity_Full']:.2f} kWh")
    
    if 'Battery_Capacity_Useable' in df1.columns:
        print(f"Battery Capacity (Useable): {max_battery_vehicle['Battery_Capacity_Useable']:.2f} kWh")
    
    # Show top 5 vehicles by battery capacity
    print("\n=== Top 5 Vehicles by Battery Capacity ===")
    top_5_battery = df1_clean.nlargest(5, 'Battery_Capacity_Full')[['Vehicle_ID', 'Vehicle_Make', 'Vehicle_Model', 'Battery_Capacity_Full']]
    print(top_5_battery.to_string(index=False))
    
    # Basic statistics
    print(f"\n=== Battery Capacity Statistics ===")
    print(f"Mean Battery Capacity: {df1_clean['Battery_Capacity_Full'].mean():.2f} kWh")
    print(f"Median Battery Capacity: {df1_clean['Battery_Capacity_Full'].median():.2f} kWh")
    print(f"Standard Deviation: {df1_clean['Battery_Capacity_Full'].std():.2f} kWh")
    print(f"Min Battery Capacity: {df1_clean['Battery_Capacity_Full'].min():.2f} kWh")
    print(f"Max Battery Capacity: {df1_clean['Battery_Capacity_Full'].max():.2f} kWh")

else:
    print("No 'Battery_Capacity_Full' column found in the dataset")
    print("Available columns:", list(df1.columns)) 