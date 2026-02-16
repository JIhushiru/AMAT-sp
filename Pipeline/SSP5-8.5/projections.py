import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
historical_path = os.path.join(data_dir, 'banana_yield_2010-2024.xlsx')
future_path = os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx')

# Clean function
def clean_yield_data(df):
    df['yield'] = df['yield'].replace('#DIV/0!', pd.NA)
    df['yield'] = pd.to_numeric(df['yield'], errors='coerce')
    df = df.dropna(subset=['yield'])
    df = df.sort_values(by='year')
    return df

# Load and clean
historical_df = clean_yield_data(pd.read_excel(historical_path))
future_df = clean_yield_data(pd.read_excel(future_path))

# ---- ANALYSIS ---- #

# 1. National Averages
national_avg_historical = historical_df['yield'].mean()
national_avg_future = future_df['yield'].mean()
national_pct_change = ((national_avg_future - national_avg_historical) / national_avg_historical) * 100

# 2. Province-Level Averages and % Change
province_avg_historical = historical_df.groupby('province')['yield'].mean()
province_avg_future = future_df.groupby('province')['yield'].mean()
province_pct_change = ((province_avg_future - province_avg_historical) / province_avg_historical) * 100

province_summary = pd.DataFrame({
    'Historical Avg (2010–2024)': province_avg_historical,
    'Future Avg (2025–2034)': province_avg_future,
    '% Change': province_pct_change
}).round(2)

# 3. 2024 vs 2034 comparison
yield_2024 = historical_df[historical_df['year'] == 2024].groupby('province')['yield'].mean()
yield_2034 = future_df[future_df['year'] == 2034].groupby('province')['yield'].mean()
compare_years = pd.DataFrame({
    '2024': yield_2024,
    '2034': yield_2034,
    '% Change': ((yield_2034 - yield_2024) / yield_2024) * 100
}).round(2)

# 4. National Trend Line
historical_trend = historical_df.groupby('year')['yield'].mean()
future_trend = future_df.groupby('year')['yield'].mean()
national_trend = pd.concat([historical_trend, future_trend]).sort_index().round(2)

# 5. Additional Count Summary
increase_2024_2034 = (compare_years['% Change'] > 0).sum()
decrease_2024_2034 = (compare_years['% Change'] < 0).sum()
increase_avg_change = (province_summary['% Change'] > 0).sum()
decrease_avg_change = (province_summary['% Change'] < 0).sum()

# ---- EXPORTS ---- #

output_path = os.path.join(base_dir, 'banana_yield_analysis.xlsx')
with pd.ExcelWriter(output_path) as writer:
    province_summary.to_excel(writer, sheet_name='Province Avg & %Change')
    compare_years.to_excel(writer, sheet_name='2024 vs 2034')
    national_trend.to_frame(name='National Avg Yield').to_excel(writer, sheet_name='National Trend')

province_summary.to_csv(os.path.join(base_dir, 'province_summary.csv'))
compare_years.to_csv(os.path.join(base_dir, 'compare_2024_2034.csv'))
national_trend.to_csv(os.path.join(base_dir, 'national_trend.csv'))

# ---- PLOTS ---- #

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
plt.plot(national_trend.index, national_trend.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'national_yield_trend.png'))
plt.close()

plt.figure(figsize=(12, 6))
sorted_prov = province_summary.sort_values('% Change', ascending=False)
sns.barplot(data=sorted_prov, x=sorted_prov.index, y='% Change', palette='coolwarm')
plt.ylabel('% Change')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'province_percent_change.png'))
plt.close()

compare_years_sorted = compare_years.dropna().sort_values('% Change', ascending=False)
compare_years_sorted[['2024', '2034']].plot(kind='bar', figsize=(12, 6))
plt.ylabel('Yield')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'yield_2024_vs_2034.png'))
plt.close()

# ---- PRINT SUMMARY ---- #
print("Analysis complete.")
print("Saved: banana_yield_analysis.xlsx, CSVs, and 3 plots:")
print("- national_yield_trend.png")
print("- province_percent_change.png")
print("- yield_2024_vs_2034.png\n")

print("Summary of Provincial Changes:")
print(f"- Provinces with yield **increase** from 2024 to 2034: {increase_2024_2034}")
print(f"- Provinces with yield **decrease** from 2024 to 2034: {decrease_2024_2034}")
print(f"- Provinces with **positive % change** (avg 2025-2034 vs 2010-2024): {increase_avg_change}")
print(f"- Provinces with **negative % change** (avg 2025-2034 vs 2010-2024): {decrease_avg_change}")
