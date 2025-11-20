import pandas as pd

# Load in the pandas dataframe
results = pd.read_csv(r"model\results\custom_predictions.csv")

# Before splitting into favored/unfavored, filter by genre(s)
def filter_by_genre(df, genres):
    valid_genres = [genre for genre in genres if genre in df.columns]
    if not valid_genres:
        raise ValueError("One or more genres in genre list were not specified in testing data.")
    
    mask = (df[genres] == 1).all(axis=1)
    
    return df[mask]

# Specify the genre to filter by
genres = ["midwest emo", "shoegaze"]
results = filter_by_genre(results, genres)

# Instantiate favored versus unfavored dataframes
favored_df = pd.DataFrame(columns=['track_name', 'track_id', 'artist_name', 'album_name', 'favor_predictions', 'favor_probabilities'])
unfavored_df = pd.DataFrame(columns=['track_name', 'track_id', 'artist_name', 'album_name', 'favor_predictions', 'favor_probabilities'])

for idx, row in results.iterrows():
    if row['favor_predictions'] == 0:
        unfavored_df = pd.concat([unfavored_df, pd.DataFrame([row[['track_name','track_id','artist_name','album_name','favor_predictions', 'favor_probabilities']]])], ignore_index=True)
    elif row['favor_predictions'] == 1:
        favored_df = pd.concat([favored_df, pd.DataFrame([row[['track_name','track_id','artist_name','album_name','favor_predictions', 'favor_probabilities']]])], ignore_index=True)
    else:
        continue

unfavored_df = unfavored_df.sort_values(by="favor_probabilities", ascending=False)
favored_df = favored_df.sort_values(by="favor_probabilities", ascending=False)

favored_base = "model/results/favored_"
unfavored_base = "model/results/unfavored_"

for genre in genres:
    favored_base = favored_base + genre + " + "
    unfavored_base = unfavored_base + genre + " + "

favored_base = favored_base.rstrip(" + ") + ".csv"
unfavored_base = unfavored_base.rstrip(" + ") + ".csv"

print("Saving results to CSV...")
favored_df.to_csv(favored_base, index=False)
unfavored_df.to_csv(unfavored_base, index=False)

#----------------------------------- END OF TECHNICAL --------------------------------------------

# results is the dataframe used
# filter_by_genre(results, ["genre"]) gives tracks by genre