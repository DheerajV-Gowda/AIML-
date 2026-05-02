import os
import pandas as pd
import pyarrow.parquet as pq

# Get the directory where the script and parquet files are located
base_dir = os.path.dirname(os.path.abspath(__file__))

months = {
    10: os.path.join(base_dir, "yellow_tripdata_2025-10.parquet"), 
    11: os.path.join(base_dir, "yellow_tripdata_2025-11.parquet"), 
    12: os.path.join(base_dir, "yellow_tripdata_2025-12.parquet")
}
results = []

for month, file in months.items():
    try:
        print(f"Processing Month {month}...")
        df = pd.read_parquet(file, engine='pyarrow')
        
        # 1. Base Time Math
        df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
        
        # 2. THE STRICT "REAL TRIP" FILTERS
        # Only Credit Card (Payment Type 1), Valid Distance, Valid Duration, Positive Money
        df_real = df[
            (df['payment_type'] == 1) & 
            (df['trip_distance'] > 0.2) & 
            (df['duration_min'] > 1) & (df['duration_min'] < 120) &
            (df['total_amount'] > 0) &
            (df['fare_amount'] > 0)
        ].copy()
        
        # 3. The Physics Filter (Speed between 3mph and 65mph)
        df_real['mph'] = df_real['trip_distance'] / (df_real['duration_min'] / 60)
        df_real = df_real[(df_real['mph'] >= 3) & (df_real['mph'] <= 65)]
        
        # Calculate Micro Metrics on the CLEANED data
        df_real['tip_ratio'] = df_real['tip_amount'] / df_real['fare_amount']
        
        congestion = df_real.get('cbd_congestion_fee', pd.Series([0]*len(df_real))).fillna(0)
        tolls = df_real['tolls_amount'].fillna(0)
        surcharge = df_real['improvement_surcharge'].fillna(0)
        
        df_real['net_yield_per_hour'] = ((df_real['total_amount'] - tolls - surcharge - congestion) / df_real['duration_min']) * 60

        results.append({
            'Month': month,
            'Real Trips (Volume)': len(df_real),
            'Strict Avg Yield/Hr ($)': df_real['net_yield_per_hour'].mean(),
            'Strict Tip Ratio (CC Only)': df_real['tip_ratio'].mean(),
            'Avg Speed (MPH)': df_real['mph'].mean()
        })
    except Exception as e:
        print(f"Error on month {month}: {e}")

print("\n--- ACT 2: THE STRICT REALITY REPORT ---")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# Export to CSV for Power BI
summary_df.to_csv(os.path.join(base_dir, 'act2_summary.csv'), index=False)
print("\nExported act2_summary.csv for Power BI.")