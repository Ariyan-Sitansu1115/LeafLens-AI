import pandas as pd

# Load your existing dataset
df = pd.read_csv("data/final1_weather_dataset.csv")

# Create new dataset
new_df = pd.DataFrame()

new_df["time"] = df["time"]
new_df["temp"] = df["tavg"]

# Fake humidity (or improve later)
new_df["humidity"] = 60 + (df["prcp"] * 2)

# Rain column
new_df["prcp"] = df["prcp"]

# Save
new_df.to_csv("data/rain_prediction_dataset.csv", index=False)

print("New dataset created!")