import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import random
from PIL import Image
import base64

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

spotify_client = os.getenv("SPOTIFY_CLIENT")
spotify_secret = os.getenv("SPOTIFY_SECRET")
spotify_redirect_uri = "http://127.0.0.1:5000/callback"

scope = "playlist-modify-public playlist-modify-private ugc-image-upload"

# Instantiate Spotipy object
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotify_client, client_secret=spotify_secret, redirect_uri=spotify_redirect_uri, scope=scope))

# Customize which file to extract data from
def csv_to_pd(path):
    df = pd.read_csv(path)
    df = df["track_id"]

    track_list = df.to_list()

    return track_list

# Extract the genre from the title of .csv
def extract_genre(path):
    genre = path[22:] # Omits prefix
    genre = genre[:-4] # Omits suffix

    print(genre)

    return genre

# Normalize upload image
def upload_image(image_path):
    base, ext = os.path.splitext(image_path)

    if ext.lower() == '.webp':
        new_path = base + '.jpg'
        image = Image.open(image_path).convert("RGB")
        image.save(new_path, "JPEG")
        image_path = new_path
    elif ext.lower() == '.png':
        new_path = base + '.jpg'
        image = Image.open(image_path).convert("RGB")
        image.save(new_path, "JPEG")
        image_path = new_path

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return image_base64

# Create playlist and add tracks
def create_playlist_add_tracks(user_id, path, number_of_tracks, image, shuffle):
    tracks = csv_to_pd(path)

    # If the user wants track entries to be shuffled before playlist addition:
    if shuffle == True:
        random.shuffle(tracks)

    tracks = tracks[0:number_of_tracks]
    
    # Extracts the genre from the csv file's name
    genre = extract_genre(path)

    # Playlist creation
    output_playlist = sp.user_playlist_create(user_id, f"[{genre}] curated by Neural Network 🎵", True, False, f"Outputs for genre {genre} from my neural network model! 🎵")
    output_playlist_id = output_playlist["id"]

    # Add playlist cover image
    sp.playlist_upload_cover_image(output_playlist_id, image)

    # Playlist track addition
    sp.user_playlist_add_tracks(user_id, output_playlist_id, tracks)

    # Print done
    print("Output playlist successfully created!")

#------------------------------------------- END OF TECHNICAL -------------------------------------

user_id = "htl4zoxjcymb38j7oe26cgpf2"

image_path = r"assets/robot.webp"
refined_image = upload_image(image_path)

# # For screamo
# screamo_path = r"model\results\favored_screamo.csv"
# create_playlist_add_tracks(user_id, screamo_path, 30, refined_image, False)

# # For plain emo
# emo_path = r"model\results\favored_emo.csv"
# create_playlist_add_tracks(user_id, emo_path, 30, refined_image, True)

# # For k-ballad
# k_ballad_path = r"model\results\favored_k-ballad.csv"
# create_playlist_add_tracks(user_id, k_ballad_path, 30, refined_image, False)

# # For skate punk
# skate_punk_path = r"model\results\favored_skate punk.csv"
# create_playlist_add_tracks(user_id, skate_punk_path, 30, refined_image, False)

# # For punk
# punk_path = r"model\results\favored_punk.csv"
# create_playlist_add_tracks(user_id, punk_path, 30, refined_image, False)

# # For math rock
# math_rock_path = r"model\results\favored_math rock.csv"
# create_playlist_add_tracks(user_id, math_rock_path, 30, refined_image, False)

# # For shoegaze
# shoegaze_path = r"model\results\favored_shoegaze.csv"
# create_playlist_add_tracks(user_id, shoegaze_path, 30, refined_image, False)

# # For shoegaze + emo
# shoegaze_emo_path = r"model\results\favored_shoegaze + emo.csv"
# create_playlist_add_tracks(user_id, shoegaze_emo_path, 30, refined_image, False)

# For midwest emo + shoegaze
midwest_emo_shoegaze_path = r"model\results\favored_midwest emo + shoegaze.csv"
create_playlist_add_tracks(user_id, midwest_emo_shoegaze_path, 30, refined_image, False)
