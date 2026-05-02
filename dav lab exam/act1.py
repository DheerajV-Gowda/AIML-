import pandas as pd
import os

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The files you uploaded (serving as proxies around the month 10-12 critical range)
files = {
    10: os.path.join(BASE_DIR, "yellow_tripdata_2025-10.parquet"),
    11: os.path.join(BASE_DIR, "yellow_tripdata_2025-11.parquet"),
    12: os.path.join(BASE_DIR, "yellow_tripdata_2025-12.parquet")
}

results = []

for month, file in files.items():
    print(f"Processing Month {month}...")
    
    # 1. Load the Parquet file
    df = pd.read_parquet(file, engine='pyarrow')
    
    # 2. Calculate actual time spent per trip (in minutes)
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    
    # Filter out extreme anomalies (e.g., zero-minute trips, data entry errors)
    df_clean = df[(df['duration_min'] > 1) & (df['duration_min'] < 120)].copy()
    df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['total_amount'] > 0)]
    
    # 3. Calculate Tip-to-Fare Ratio (Are passengers tipping less to offset fees?)
    df_clean['tip_to_fare_ratio'] = df_clean['tip_amount'] / df_clean['fare_amount']
    
    # 4. Calculate Net Driver Yield per Hour (The true measure of efficiency)
    # Deducting surcharges, tolls, and the new 2025 CBD congestion fee
    congestion_fee = df_clean['cbd_congestion_fee'].fillna(0) if 'cbd_congestion_fee' in df_clean.columns else 0
    surcharge = df_clean['improvement_surcharge'].fillna(0) if 'improvement_surcharge' in df_clean.columns else 0
    tolls = df_clean['tolls_amount'].fillna(0) if 'tolls_amount' in df_clean.columns else 0
    
    df_clean['net_yield'] = df_clean['total_amount'] - tolls - surcharge - congestion_fee
    
    # Normalize the take-home pay to an hourly rate based on trip duration
    df_clean['net_yield_per_hour'] = (df_clean['net_yield'] / df_clean['duration_min']) * 60

    # 5. Aggregate the final monthly metrics
    results.append({
        'Month': month,
        'Total Trip Volume': len(df_clean),
        'Macro Total Revenue ($)': df_clean['total_amount'].sum(),
        'Avg Trip Duration (min)': df_clean['duration_min'].mean(),
        'Avg Driver Net Yield/Hr ($)': df_clean['net_yield_per_hour'].mean(),
        'Avg Tip-to-Fare Ratio': df_clean['tip_to_fare_ratio'].mean()
    })

# Display the findings
summary_df = pd.DataFrame(results)
print("\n--- INEFFICIENCY DIAGNOSTIC REPORT ---")
print(summary_df.to_string(index=False))

# Export to CSV for Power BI
summary_df.to_csv(os.path.join(BASE_DIR, 'act1_summary.csv'), index=False)
print("\nExported act1_summary.csv for Power BI.")