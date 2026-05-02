import pandas as pd
import os

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The HVFHV files (High Volume For-Hire Vehicle - Uber/Lyft)
months = {
    10: os.path.join(BASE_DIR, "fhvhv_tripdata_2025-10.parquet"), 
    11: os.path.join(BASE_DIR, "fhvhv_tripdata_2025-11.parquet"), 
    12: os.path.join(BASE_DIR, "fhvhv_tripdata_2025-12.parquet")
}

results = []

for month, file in months.items():
    print(f"Processing HVFHV Month {month}...")
    try:
        # Load the Uber/Lyft data
        df = pd.read_parquet(file, engine='pyarrow')
        
        # 1. Base Time Math (Column names are slightly different for HVFHV)
        df['duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60.0
        
        # 2. Filter for valid, completed trips
        df_clean = df[
            (df['trip_miles'] > 0.2) & 
            (df['duration_min'] > 1) & (df['duration_min'] < 120) &
            (df['base_passenger_fare'] > 0)
        ].copy()
        
        # 3. Calculate Macro Revenue (What the passenger actually paid in total)
        tolls = df_clean['tolls'].fillna(0)
        sales_tax = df_clean['sales_tax'].fillna(0)
        congestion = df_clean['congestion_surcharge'].fillna(0)
        tips = df_clean['tips'].fillna(0)
        base_fare = df_clean['base_passenger_fare']
        
        df_clean['total_revenue'] = base_fare + tolls + sales_tax + congestion + tips
        
        # 4. Calculate The Falsifiability Metric: Tip-to-Fare Ratio
        df_clean['tip_ratio'] = tips / base_fare
        
        # 5. Calculate Driver Yield per Hour (Using the explicit driver_pay column)
        df_clean['driver_yield_per_hour'] = (df_clean['driver_pay'] / df_clean['duration_min']) * 60
        
        # Aggregate the final monthly metrics
        results.append({
            'Month': month,
            'Total HVFHV Trips': len(df_clean),
            'Macro Total Revenue ($)': df_clean['total_revenue'].sum(),
            'Avg Driver Pay/Hr ($)': df_clean['driver_yield_per_hour'].mean(),
            'Avg Tip Ratio (%)': df_clean['tip_ratio'].mean() * 100
        })
        
    except Exception as e:
        print(f"Error on month {month}: {e}")

# Display the final verification
summary_df = pd.DataFrame(results)
print("\n--- THE FALSIFIABILITY TEST: UBER/LYFT (HVFHV) DATA ---")
print(summary_df.to_string(index=False))

# Export to CSV for Power BI
summary_df.to_csv(os.path.join(BASE_DIR, 'act3_summary.csv'), index=False)
print("\nExported act3_summary.csv for Power BI.")