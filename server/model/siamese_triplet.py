'''
Docstring for server.model.siamese_triplet

***OBJECTIVE***
1. Manually create classes that separate dissimilar genres and group genres that are similar.
2. Create a program that creates triplets of songs
    i. Each song should appear at least once (as the anchor)
    ii. Each song other than the anchor will be assigned a positive or negative value based on
        class proximity
    iii. No triplet has to be absolutely perfect; the imperfection gives it the way to learn
3. Develop a siamese model based on the positive/negative weights
'''

import csv
import json
import pandas as pd

genre_classes_path = r"server\model\genre_classifications.json"
dataset_path = r"server\model\music_dataset.csv"
seed_classes_path = r"server\model\seed_classifications.json"

# Load in the genre classifications
with open(genre_classes_path, mode="r") as f:
    genre_classes = json.load(f)

# Load in the seed classifications
with open(seed_classes_path, mode="r") as f:
    seed_classes = json.load(f)

# Load in the song dataset information
df = pd.read_csv(dataset_path)

'''
Triplet Generation:

Compare each song based on the following traits (priority decreases with order):
1. VAD: Determine a threshold for this
2. Energy & Intensity: Use RMS mean/std, spectral centroid & contrast, and ZCR
3. Timbre & Texture: MFCC distributions, spectral flatness & rolloff
4. Seed Reinforcement: If the song is within the same seed class, give it higher positive rating
'''

import numpy as np

audio_data_path = r"server\model\dataset\audio_data.npy"

# Load in the auditory data
audio_tensors = np.load(audio_data_path)

# print(f"Test: {audio_tensors[0]}") # Works

def split_data(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]

batches = list(split_data(audio_tensors, 502))

print(f"Done: {batches}")