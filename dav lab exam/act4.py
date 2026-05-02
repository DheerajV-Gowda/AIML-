import pandas as pd
import numpy as np
import os

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Extracting the City's Rhythm...")

# Load December (The epicenter of our analysis)
file = os.path.join(BASE_DIR, "yellow_tripdata_2025-12.parquet")
df = pd.read_parquet(file, engine='pyarrow')

# 1. Temporal Transformation
df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['hour'] = df['pickup_dt'].dt.hour
df['day_of_week'] = df['pickup_dt'].dt.dayofweek # 0=Monday, 6=Sunday

# 2. Recreate the Strict Filter
df['duration_min'] = (df['tpep_dropoff_datetime'] - df['pickup_dt']).dt.total_seconds() / 60.0
strict_mask = (
    (df['payment_type'] == 1) & 
    (df['trip_distance'] > 0.2) & 
    (df['duration_min'] > 1) & (df['duration_min'] < 120) &
    (df['total_amount'] > 0)
)

# 3. Aggregate by the 168-Hour Week
# Get Raw Data (The Illusion)
raw_hourly = df.groupby(['day_of_week', 'hour']).agg(
    raw_trips=('total_amount', 'count'),
    raw_tips=('tip_amount', 'sum'),
    raw_fares=('fare_amount', 'sum')
).reset_index()
raw_hourly['raw_tip_ratio'] = raw_hourly['raw_tips'] / raw_hourly['raw_fares']

# Get Strict Data (The Reality)
df_strict = df[strict_mask]
strict_hourly = df_strict.groupby(['day_of_week', 'hour']).agg(
    strict_trips=('total_amount', 'count'),
    strict_tips=('tip_amount', 'sum'),
    strict_fares=('fare_amount', 'sum')
).reset_index()
strict_hourly['strict_tip_ratio'] = strict_hourly['strict_tips'] / strict_hourly['strict_fares']

# 4. Merge to find the Rhythm
rhythm = pd.merge(raw_hourly, strict_hourly, on=['day_of_week', 'hour'])
rhythm['evasion_gap'] = rhythm['raw_trips'] - rhythm['strict_trips']
rhythm['evasion_rate_pct'] = (rhythm['evasion_gap'] / rhythm['raw_trips']) * 100

# 5. Observe the Extremes (Is it stable, or is it cyclical?)
wed_peak = rhythm[(rhythm['day_of_week'] == 2) & (rhythm['hour'] == 17)].iloc[0] # Wed 5 PM
sat_night = rhythm[(rhythm['day_of_week'] == 5) & (rhythm['hour'] == 2)].iloc[0] # Sat 2 AM

print("\n--- THE TEMPORAL ANATOMY OF EVASION (DEC 2025) ---")
print("SCENARIO A: The Gridlock Pulse (Wednesday, 5:00 PM)")
print(f"  Raw Tip Ratio (Illusion): {wed_peak['raw_tip_ratio']*100:.1f}%")
print(f"  Strict CC Tip (Reality):  {wed_peak['strict_tip_ratio']*100:.1f}%")
print(f"  System Evasion Rate:      {wed_peak['evasion_rate_pct']:.1f}% off-the-books\n")

print("SCENARIO B: The Open Road (Saturday, 2:00 AM)")
print(f"  Raw Tip Ratio (Illusion): {sat_night['raw_tip_ratio']*100:.1f}%")
print(f"  Strict CC Tip (Reality):  {sat_night['strict_tip_ratio']*100:.1f}%")
print(f"  System Evasion Rate:      {sat_night['evasion_rate_pct']:.1f}% off-the-books")

# Export to CSV for Power BI
rhythm.to_csv(os.path.join(BASE_DIR, 'act4_summary.csv'), index=False)
print("\nExported act4_summary.csv for Power BI.")