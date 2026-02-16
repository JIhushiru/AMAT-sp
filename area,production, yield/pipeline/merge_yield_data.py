import os
import pandas as pd
import re


def clean_province_name(name):
    """Clean province names by removing dots and trimming whitespace"""
    return re.sub(r"^\.+", "", name).strip()


def main():
    # Get the current directory
    current_dir = os.getcwd()

    # Define file paths
    yield_file = os.path.join(current_dir, "banana_yield_2010-2024.csv")
    data_file = os.path.join(current_dir, "banana_2010-2023.csv")
    output_file = os.path.join(current_dir, "banana_2010-2023_updated.csv")

    # Read the yield data (horizontal format)
    yield_df = pd.read_csv(yield_file)

    # Read the main data file (vertical format)
    data_df = pd.read_csv(data_file)

    # Create a dictionary to store province-year-yield mappings
    yield_dict = {}

    # Process the yield dataframe to create mappings
    for _, row in yield_df.iterrows():
        province = clean_province_name(row["Geolocation"])

        # Skip the national data (PHILIPPINES)
        if province == "PHILIPPINES" or "REGION" in province:
            continue

        # Create mappings for each year column
        for year in range(
            2010, 2024
        ):  # Only using 2010-2023 as that's what's in our target file
            year_col = f"{year} Annual"
            if year_col in row:
                # Store yield value with province and year as key
                yield_dict[(province.upper(), year)] = row[year_col]

    # Function to map province and year to yield value
    def get_yield(row):
        # Extract the province name from the first non-numeric column
        province = row["province"].upper()
        year = row["year"]

        # Look up the yield value
        key = (province, year)
        if key in yield_dict:
            return yield_dict[key]
        return row["yield"]  # Keep original if not found

    # Apply the mapping to update yield values
    data_df["yield"] = data_df.apply(get_yield, axis=1)

    # Save the updated dataframe
    data_df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")


if __name__ == "__main__":
    main()
