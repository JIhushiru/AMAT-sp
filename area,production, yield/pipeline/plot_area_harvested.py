import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "banana_area_harvested_2010-2024.csv")

# Load the data
try:
    data = pd.read_csv(csv_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    exit()

# Extract Philippines' data
philippines_data = data.iloc[0, 2:].astype(float)
years = philippines_data.index.str.replace(" Annual", "").astype(int)

# Calculate statistics
mean_area = philippines_data.mean()
initial = philippines_data.iloc[0]
final = philippines_data.iloc[-1]
percent_change = ((final - initial) / initial) * 100

# Create plot
plt.figure(figsize=(12, 6))

# Main plot with connecting lines
plt.plot(
    years,
    philippines_data,
    marker="o",
    linestyle="-",
    color="green",
    linewidth=2,
    markersize=8,
    label="Harvest Area",
)

# Mean line
plt.axhline(
    y=mean_area,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {mean_area:,.0f} ha",
)

# 2010-2024 trend line
plt.plot(
    [2010, 2024],
    [initial, final],
    color="blue",
    linestyle="--",
    linewidth=1.5,
    label=f"Start-End Trend (+{percent_change:.2f}%)",
)

# Formatting
plt.title("Philippines Banana Harvest Area (2010-2024)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Area (hectares)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
