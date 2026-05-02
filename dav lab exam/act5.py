import pandas as pd
import numpy as np
import os

# Baseline System States (From Act 3 Findings)
hours = np.arange(0, 24)
# Corporate compliance peaks during the day, drops at night
corporate_ratio = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.8, 0.85, 0.9, 0.85, 0.8, 0.75, 
                            0.7, 0.7, 0.75, 0.8, 0.85, 0.85, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1])
base_evasion = 1.0 - corporate_ratio

# THE INTERVENTION: Seed 11
fare_increase_pct = 0.11  # 11% hike

# The Behavioral Rules (Elasticity)
# Corporate riders absorb the cost (0 evasion increase)
# Non-corporate riders are highly sensitive: a 1% fare increase = 1.5% evasion increase
evasion_multiplier = 1.0 + (fare_increase_pct * 1.5) 

# Simulate the Shift
simulated_evasion = np.clip(base_evasion * evasion_multiplier, 0, 0.95)

# Calculate the Delta
results = pd.DataFrame({
    'Hour': hours,
    'Base_Evasion_Rate': base_evasion * 100,
    'Simulated_Evasion_Rate': simulated_evasion * 100
})
results['Evasion_Contagion_Delta'] = results['Simulated_Evasion_Rate'] - results['Base_Evasion_Rate']

# Display the Tipping Points
print("--- THE 11% POLICY SHOCK SIMULATION ---")
print(f"Daytime (Hour 14 / 2 PM) - Corporate Absorbs: +{results.loc[14, 'Evasion_Contagion_Delta']:.1f}% evasion shift")
print(f"Evening (Hour 19 / 7 PM) - Contagion Begins:  +{results.loc[19, 'Evasion_Contagion_Delta']:.1f}% evasion shift")
print(f"Late Night (Hour 2 / 2 AM) - Complete Break:   +{results.loc[2, 'Evasion_Contagion_Delta']:.1f}% evasion shift")

# Export to CSV for Power BI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results.to_csv(os.path.join(BASE_DIR, 'act5_summary.csv'), index=False)
print("\nExported act5_summary.csv for Power BI.")