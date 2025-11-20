import kagglehub
import pandas as pd

# Download latest version
# path = kagglehub.dataset_download("cakiki/muse-the-musical-sentiment-dataset")

# print("Path to dataset files:", path)

# path_to_csv = r"server\sentiment_model\muse_v3.csv"

# print("Turning into df")
# kaggle_df = pd.read_csv(path_to_csv)

# kaggle_df = kaggle_df[kaggle_df["spotify_id"].notna() & (kaggle_df["spotify_id"] != "")]

# kaggle_df.to_csv(path_to_csv)

import csv
import json

csv_file = r"server\sentiment_model\muse_v3.csv"
json_file = r"server\sentiment_model\muse_v3.json"

data = []

with open(csv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)
    
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)