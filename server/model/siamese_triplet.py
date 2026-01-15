"""
Docstring for server.model.siamese_triplet

***OBJECTIVE***
1. Manually create classes that separate dissimilar genres and group genres that are similar.
2. Create a program that creates triplets of songs
    i. Each song should appear at least once (as the anchor)
    ii. Each song other than the anchor will be assigned a positive or negative value based on
        class proximity
    iii. No triplet has to be absolutely perfect; the imperfection gives it the way to learn
3. Develop a siamese model based on the positive/negative weights
"""

import csv
import json
import pandas as pd
import numpy as np

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

"""
Data Normalization:

For all data that are measured on different scales, we should normalize them so that data that is
measured on larger scales do not become more favorable compared to data that is measured on smaller
scales. For the auditory tensors, z-score normalization seems to be the most appropriate since the
data is purely statistical. For the semantic data, min-max normalization seems to be the most
appropriate since we need to apply the right weights to each of the sentiment tags.

Algorithm:
1. Create a dataframe from the auditory tensors
2. Iterate through each column
3. Depending on semantics or auditory data, we will need to follow the correct normalization:
    i. Min-Max Normalization:
        X' = (X - X_min) / X_max - X_min
    ii. Z-Score Normalization:
        Z = (X - /mu) / /sigma
"""

def normalize():
    semantic_csv_path = r"server\model\dataset\semantic_data.csv"
    audio_csv_path = r"server\model\dataset\pre_audio_data.csv"

    semantic_df = pd.read_csv(semantic_csv_path)
    audio_df = pd.read_csv(audio_csv_path)

    ### Audio Normalization ###
    audio_cols_to_norm = [
        "tempo",
        "tempo_bt",
        "rms_mean",
        "rms_std",
        "rms_min",
        "rms_max",
        "zcr_mean",
        "zcr_std",
        "zcr_min",
        "zcr_max",
        "spec_centroid_mean",
        "spec_centroid_std",
        "spec_centroid_min",
        "spec_centroid_max",
        "spec_bandwidth_mean",
        "spec_bandwidth_std",
        "spec_bandwidth_min",
        "spec_bandwidth_max",
        "spec_contrast_1_mean",
        "spec_contrast_1_std",
        "spec_contrast_1_min",
        "spec_contrast_1_max",
        "spec_contrast_2_mean",
        "spec_contrast_2_std",
        "spec_contrast_2_min",
        "spec_contrast_2_max",
        "spec_contrast_3_mean",
        "spec_contrast_3_std",
        "spec_contrast_3_min",
        "spec_contrast_3_max",
        "spec_contrast_4_mean",
        "spec_contrast_4_std",
        "spec_contrast_4_min",
        "spec_contrast_4_max",
        "spec_contrast_5_mean",
        "spec_contrast_5_std",
        "spec_contrast_5_min",
        "spec_contrast_5_max",
        "spec_contrast_6_mean",
        "spec_contrast_6_std",
        "spec_contrast_6_min",
        "spec_contrast_6_max",
        "spec_contrast_7_mean",
        "spec_contrast_7_std",
        "spec_contrast_7_min",
        "spec_contrast_7_max",
        "spec_flatness_mean",
        "spec_flatness_std",
        "spec_flatness_min",
        "spec_flatness_max",
        "spec_rolloff_mean",
        "spec_rolloff_std",
        "spec_rolloff_min",
        "spec_rolloff_max",
        "mfcc_1_mean",
        "mfcc_1_std",
        "mfcc_1_min",
        "mfcc_1_max",
        "mfcc_2_mean",
        "mfcc_2_std",
        "mfcc_2_min",
        "mfcc_2_max",
        "mfcc_3_mean",
        "mfcc_3_std",
        "mfcc_3_min",
        "mfcc_3_max",
        "mfcc_4_mean",
        "mfcc_4_std",
        "mfcc_4_min",
        "mfcc_4_max",
        "mfcc_5_mean",
        "mfcc_5_std",
        "mfcc_5_min",
        "mfcc_5_max",
        "mfcc_6_mean",
        "mfcc_6_std",
        "mfcc_6_min",
        "mfcc_6_max",
        "mfcc_7_mean",
        "mfcc_7_std",
        "mfcc_7_min",
        "mfcc_7_max",
        "mfcc_8_mean",
        "mfcc_8_std",
        "mfcc_8_min",
        "mfcc_8_max",
        "mfcc_9_mean",
        "mfcc_9_std",
        "mfcc_9_min",
        "mfcc_9_max",
        "mfcc_10_mean",
        "mfcc_10_std",
        "mfcc_10_min",
        "mfcc_10_max",
        "mfcc_11_mean",
        "mfcc_11_std",
        "mfcc_11_min",
        "mfcc_11_max",
        "mfcc_12_mean",
        "mfcc_12_std",
        "mfcc_12_min",
        "mfcc_12_max",
        "mfcc_13_mean",
        "mfcc_13_std",
        "mfcc_13_min",
        "mfcc_13_max",
        "chromagram_1_mean",
        "chromagram_1_std",
        "chromagram_1_min",
        "chromagram_1_max",
        "chromagram_2_mean",
        "chromagram_2_std",
        "chromagram_2_min",
        "chromagram_2_max",
        "chromagram_3_mean",
        "chromagram_3_std",
        "chromagram_3_min",
        "chromagram_3_max",
        "chromagram_4_mean",
        "chromagram_4_std",
        "chromagram_4_min",
        "chromagram_4_max",
        "chromagram_5_mean",
        "chromagram_5_std",
        "chromagram_5_min",
        "chromagram_5_max",
        "chromagram_6_mean",
        "chromagram_6_std",
        "chromagram_6_min",
        "chromagram_6_max",
        "chromagram_7_mean",
        "chromagram_7_std",
        "chromagram_7_min",
        "chromagram_7_max",
        "chromagram_8_mean",
        "chromagram_8_std",
        "chromagram_8_min",
        "chromagram_8_max",
        "chromagram_9_mean",
        "chromagram_9_std",
        "chromagram_9_min",
        "chromagram_9_max",
        "chromagram_10_mean",
        "chromagram_10_std",
        "chromagram_10_min",
        "chromagram_10_max",
        "chromagram_11_mean",
        "chromagram_11_std",
        "chromagram_11_min",
        "chromagram_11_max",
        "chromagram_12_mean",
        "chromagram_12_std",
        "chromagram_12_min",
        "chromagram_12_max",
    ]

    means = audio_df[audio_cols_to_norm].mean()
    stds = audio_df[audio_cols_to_norm].std()

    audio_index = audio_df.copy()
    audio_index = audio_index.drop(audio_cols_to_norm, axis=1)

    audio_df_norm = (audio_df[audio_cols_to_norm] - means) / stds

    audio_df_full = pd.concat([audio_index, audio_df_norm], axis=1)


    ### Semantic Normalization ###
    cols_to_norm = ["valence_tags", "arousal_tags", "dominance_tags"]

    # X' = (X - X_min) / X_max - X_min
    semantic_df[cols_to_norm] = (
        semantic_df[cols_to_norm] - semantic_df[cols_to_norm].min()
    ) / (semantic_df[cols_to_norm].max() - semantic_df[cols_to_norm].min())

    normalized_semantic_path = r"server\model\dataset\normalized_semantic.csv"
    normalized_audio_path = r"server\model\dataset\normalized_audio.npy"

    semantic_df.to_csv(normalized_semantic_path)

    audio_array = audio_df_full.to_numpy()
    np.save(normalized_audio_path, audio_array)

    print("Data has been normalized!")

"""
Triplet Generation:

Compare each song based on the following traits (priority decreases with order):
1. VAD: Determine a threshold for this
2. Energy & Intensity: Use RMS mean/std, spectral centroid & contrast, and ZCR
3. Timbre & Texture: MFCC distributions, spectral flatness & rolloff
4. Seed Reinforcement: If the song is within the same seed class, give it higher positive rating
"""

audio_data_path = r"server\model\dataset\normalized_audio.npy"

# Load in the auditory data
audio_tensors = np.load(audio_data_path)

# print(f"Test: {audio_tensors[0]}") # Works

def split_data(data, n):
    for i in range(0, len(data), n):
        yield data[i : i + n]


batches = list(
    split_data(audio_tensors, 502)
)  # Make batches 502 so we only have 4 songs left over

print(f"Done: {batches}")