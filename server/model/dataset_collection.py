"""
Steps to take:

1. Get a dataset (100 liked songs, 100 songs that are of varied genres that I don't like/don't listen to)
2. Refine the data
3. 
"""

import os
import csv
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import time

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

spotify_client = os.getenv("SPOTIFY_CLIENT")
spotify_secret = os.getenv("SPOTIFY_SECRET")
spotify_redirect_uri = "http://127.0.0.1:5000/callback"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotify_client, client_secret=spotify_secret, redirect_uri=spotify_redirect_uri))

"""
Things I need for data:

1. Song title
2. Song genre
3. Song popularity
4. Song duration
5. Release year

"""
positive_dataset = sp.playlist_tracks("6uNsxkl68cNqFticzeiOBP", fields="items(track(artists(id, name), album(name, release_date), id, disc_number, duration_ms, name, popularity))")["items"]
negative_dataset = sp.playlist_tracks("56toYCBmVhzrlxuogdJm3O", fields="items(track(artists(id, name), album(name, release_date), id, disc_number, duration_ms, name, popularity))")["items"]
# print(test)

def fetch_all_tracks(sp, playlist_id):
    print("Starting fetch...")
    time.sleep(0.1)
    results = sp.playlist_tracks(playlist_id, fields="items(track(artists(id, name), album(name, release_date), id, disc_number, duration_ms, name, popularity)), next")

    tracks = [item['track'] for item in results['items']]

    while results.get('next'):
        results = sp.next(results)
        tracks.extend([item['track'] for item in results['items']])

    return tracks

positive_dataset = fetch_all_tracks(sp, playlist_id="6uNsxkl68cNqFticzeiOBP")
negative_dataset = fetch_all_tracks(sp, playlist_id="56toYCBmVhzrlxuogdJm3O")

# Normalizing duration
max_duration = 7 * 60 * 1000


# Extract the track
# track = track_data['track']

mhe_genres_dict = []

print("Beginning multi-hot encoding process...")
# GENRE COLLECTING FUNCTION
def mhe_genre_collection(dataset):
    for track_data in dataset:
        time.sleep(0.1)
        # Extract the track
        track = track_data

        # Get Artist and run separate extraction
        artist_id = track['artists'][0]['id']
        artist = sp.artist(artist_id=artist_id)

        # Artist metadata:
        artist_name = artist['name']
        genres = artist['genres'] # Correct

        if len(genres) == 0:
            continue
        else:
            for genre in genres:
                if genre in mhe_genres_dict:
                    continue
                else:
                    mhe_genres_dict.append(genre)
                    print(f"Appended! {genre}")


        print(artist_name)
    print("Done\n")
    
# POSITIVE DATASET FEATURE EXTRACTION
mhe_genre_collection(positive_dataset)

# NEGATIVE DATASET FEATURE EXTRACTION
mhe_genre_collection(negative_dataset)

# print(mhe_genres_dict)
genre_to_index = {genre: i for i, genre in enumerate(mhe_genres_dict)}

# print(genre_to_index)

# All genres
all_genres = ['slowcore', 'grindcore', 'sludge metal', 'alternative rock', 'rock', 'art rock',
               'shoegaze', 'k-ballad', 'k-rap', 'k-pop', 'deathcore', 'death metal', 'screamo',
               'midwest emo', 'math rock', 'thai rock', 'thai pop', 'phleng phuea chiwit',
               't-pop', 'dream pop', 'pop punk', 'emo', 'punk', 'alternative metal', 'nu metal',
               'post-grunge', 'experimental hip hop', 'east coast hip hop', 'jazz rap',
               'alternative hip hop', 'old school hip hop', 'hip hop', 'skate punk', 'modern blues',
               'blues rock', 'blues', 'southern rock', 'metal', 'rap metal', 'anti-folk', 'folk punk',
               'riot grrrl', 'kayokyoku', 'j-pop', 'k-rock', 'post-hardcore', 'mandopop', 'c-pop',
               'taiwanese pop', 'j-rock', 'gospel', 'japanese indie', 'post-rock', 'space rock', 'classical',
               'neoclassical', 'orchestral', 'classical piano', 'chamber music', 'indie', 'metalcore',
               'christian hip hop', 'emo rap', 'horrorcore', 'cloud rap', 'trap metal', 'underground hip hop',
               'indie punk']

# Developing multi-hot encoding (MHE)
def mhe_encoding(track_genres, genre_to_index):
    vector = np.zeros(len(genre_to_index))
    for genre in track_genres:
        if genre in genre_to_index:
            vector[genre_to_index[genre]] = 1
    return vector

# CSV Headers
csv_headers = [
    'track_name', 'track_id', 'duration', 'track_popularity', 'disc_number', 'artist_name', 'artist_popularity', 
    'album_name', 'album_year', 'favor'
] + mhe_genres_dict

# Open CSV
datasets = [positive_dataset, negative_dataset]

favor = 0

def build_csv(datasets):
    # Starting training/testing CSV...
    with open ('spotify_dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        count = 0

        for dataset in datasets:
            if count == 0:
                favor = 1
            elif count == 1:
                favor = 0

            for track_data in dataset:
                track = track_data
                
                artist_id = track['artists'][0]['id']
                artist = sp.artist(artist_id=artist_id)

                row = {
                    'track_name': track['name'],
                    'track_id': track['id'],
                    'duration': (track['duration_ms'] / max_duration),
                    'track_popularity': (track['popularity'] / 100),
                    'disc_number': track['disc_number'],
                    'artist_name': artist['name'],
                    'artist_popularity': (artist['popularity'] / 100),
                    'album_name': track['album']['name'],
                    'album_year': int(track['album']['release_date'][0:4]),
                    'favor': favor
                }

                track_genres = artist['genres']

                for genre in mhe_genres_dict:
                    row[genre] = 1 if genre in track_genres else 0
                
                writer.writerow(row)
            
            count += 1

    print("Successfully exported to csv!")

build_csv(datasets=datasets)
custom_dataset = sp.playlist_tracks("4ztoPU0BaECDMRBMJAc7Sb", fields="items(track(artists(id, name), album(name, release_date), disc_number, duration_ms, name, popularity))")["items"]

all_tracks = fetch_all_tracks(sp, playlist_id="4ztoPU0BaECDMRBMJAc7Sb")
print(f"Length of tracks: {len(all_tracks)}")

def build_custom_csv(dataset):
    time.sleep(0.1)
    print("Starting custom CSV build")
    with open ('custom_spotify_dataset1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for track_data in dataset:
            track = track_data

            duration = track['duration_ms']
            
            # If the track's norm duration is greater than 0, skip
            if (duration / max_duration) > 1:
                continue
            
            artist_id = track['artists'][0]['id']
            artist = sp.artist(artist_id=artist_id)

            row = {
                'track_name': track['name'],
                'track_id': track['id'],
                'duration': (track['duration_ms'] / max_duration),
                'track_popularity': (track['popularity'] / 100),
                'disc_number': track['disc_number'],
                'artist_name': artist['name'],
                'artist_popularity': (artist['popularity'] / 100),
                'album_name': track['album']['name'],
                'album_year': int(track['album']['release_date'][0:4]),
            }

            track_genres = artist['genres']

            for genre in mhe_genres_dict:
                row[genre] = 1 if genre in track_genres else 0
            
            writer.writerow(row)

    print("Successfully exported to csv!")

build_custom_csv(all_tracks)