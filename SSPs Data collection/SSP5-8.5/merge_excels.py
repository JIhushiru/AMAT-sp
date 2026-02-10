import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

excel_path = os.path.join(base_dir, 'banana_yield_2010-2024.xlsx')
xlsx_path = os.path.join(base_dir, 'province_year_all_features_2025_2034.xlsx')

avg_df = pd.read_excel(excel_path)
ssp_df = pd.read_excel(xlsx_path)

required_features = ['cld', 'wet', 'vap', 'tmx', 'tmp', 'tmn', 'pre', 'pet', 'dtr', 'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws']
assert all(col in avg_df.columns for col in required_features), "Missing columns in average data"
assert all(col in ssp_df.columns for col in ['province', 'year', 'pre', 'tmp']), "Missing required columns in SSP data"

years = list(range(2025, 2035))
avg_by_province = avg_df.groupby('province')[required_features].mean().reset_index()

repeated_avg = pd.DataFrame([
    dict(province=row['province'], year=year, **{feat: row[feat] for feat in required_features})
    for _, row in avg_by_province.iterrows()
    for year in years
])

merged = repeated_avg.merge(ssp_df[['province', 'year', 'pre', 'tmp']],
                            on=['province', 'year'],
                            how='left',
                            suffixes=('', '_ssp'))

merged['pre'] = merged['pre_ssp']
merged['tmp'] = merged['tmp_ssp']
merged = merged.drop(columns=['pre_ssp', 'tmp_ssp'])

output_path = os.path.join(base_dir, 'merged_future_data.csv')
merged.to_csv(output_path, index=False)

print(f"Merged data saved to: {output_path}")
