import ee
import datetime

ee.Initialize(project='sp-final-project-459621')

# === Configuration ===
region = ee.Geometry.Rectangle([116.0, 4.5, 127.0, 21.0])  # Philippines
features = ["pr", "tas"]
model = "GFDL-ESM4"
scenario = "ssp245"
start_year = 2025
end_year = 2034
scale = 25000

# === Generate monthly ranges ===
months = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end_dt = datetime.datetime(year, month, 1) + datetime.timedelta(days=32)
        end = end_dt.replace(day=1).strftime('%Y-%m-%d')
        months.append((start, end))

# === Process each feature ===
for feature in features:
    print(f"\nProcessing feature: {feature}")
    collection = ee.ImageCollection("NASA/GDDP-CMIP6") \
        .filter(ee.Filter.eq("model", model)) \
        .filter(ee.Filter.eq("scenario", scenario)) \
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31") \
        .select(feature)

    print(f"  Total images in collection: {collection.size().getInfo()}")

    monthly_images = []
    for start, end in months:
        month_collection = collection.filterDate(start, end)

        if feature == 'tas':
            monthly = month_collection.mean().subtract(273.15).set("system:time_start", ee.Date(start).millis())
        elif feature == 'pr':
            seconds_in_month = 30 * 24 * 60 * 60
            monthly = month_collection.mean().multiply(seconds_in_month).set("system:time_start", ee.Date(start).millis())

        monthly = monthly.rename(f"{feature}_{start[:7]}")
        monthly_images.append(monthly)

    multi_band_image = ee.Image.cat(monthly_images)
    print(f"  Number of monthly images for {feature}: {len(monthly_images)}")

    task = ee.batch.Export.image.toDrive(
        image=multi_band_image,
        description=f'GFDL_{feature}_monthly_{start_year}_{end_year}_ssp245',
        folder='EarthEngineExports',
        fileNamePrefix=f'gfdl_{feature}_{start_year}_{end_year}_monthly',
        region=region,
        scale=scale,
        maxPixels=1e13
    )
    task.start()
    print(f"  Export task started for {feature}. Check Tasks tab.")
