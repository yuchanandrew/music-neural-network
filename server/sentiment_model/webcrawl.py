import json
import time
import copy

# ------------------------------------------ WEBCRAWLER --------------------------------------------

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_argument("--log-level=3")

import pyaudio
import wave
import os

def record_song():
    # Delete old file
    if os.path.exists(r"server\sentiment_model\output.wav"):
        os.remove(r"server\sentiment_model\output.wav")

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 15
    DEVICE_INDEX = 3

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=DEVICE_INDEX
    )

    frames = []

    print("Recording...")

    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Pyaudio done.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("server/sentiment_model/output.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

def open_selenium():
    driver = webdriver.Chrome()
    link_to_crawl = "http://localhost:5173/"

    wait = WebDriverWait(driver, 20)
    driver.get(link_to_crawl)

    wait.until(
        EC.visibility_of_all_elements_located((By.TAG_NAME, "img"))
    )

    elem = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[@id='target']"))
    )

    time.sleep(2)

    print(f"Elem: {elem}")

    elem.click()

    record_song()
    
    time.sleep(1)

    driver.quit()
    
    print("Selenium done.")

# open_selenium()

# -------------------------------------- LIBROSA EXTRACTION --------------------------------------

import librosa as lib
import numpy as np

# Normalize all features into the same shape
def summarize(feature):
    feature = np.nan_to_num(feature)
    summary_array = []

    for i in range (feature.shape[0]):
        band = feature[i]

        summary_array.append(
            [
                float(np.mean(band)),
                float(np.std(band)),
                float(np.min(band)),
                float(np.max(band))
            ]
        )
    
    return summary_array

def assign_summary(json, feature, prefix):
    summary_array = summarize(feature)
    # For vectors
    if feature.shape[0] == 1:
        mean, std, min, max = summary_array[0]
        json[f"{prefix}_mean"] = mean
        json[f"{prefix}_std"] = std
        json[f"{prefix}_min"] = min
        json[f"{prefix}_max"] = max

    # For matrices
    elif feature.shape[0] > 1:
        for i, (mean, std, min, max) in enumerate(summary_array, start=1):
            json[f"{prefix}_{i}_mean"] = mean
            json[f"{prefix}_{i}_std"] = std
            json[f"{prefix}_{i}_min"] = min
            json[f"{prefix}_{i}_max"] = max

def analyze_audio(audio_path, input_json_path, output_json_path, json_index):

    ### Audio Analysis ###

    y, sr = lib.load(audio_path)

    onset_env = lib.onset.onset_strength(y=y, sr=sr)
    tempo = lib.feature.tempo(onset_envelope=onset_env, sr=sr)
    rms = lib.feature.rms(y=y) # Loudness
    zcr = lib.feature.zero_crossing_rate(y=y) # Noisiness

    spec_centroid = lib.feature.spectral_centroid(y=y, sr=sr) # This and next 4 are on the spectogram shape
    spec_bandwidth = lib.feature.spectral_bandwidth(y=y, sr=sr)
    spec_contrast = lib.feature.spectral_contrast(y=y, sr=sr)
    spec_flatness = lib.feature.spectral_flatness(y=y)
    spec_rolloff = lib.feature.spectral_rolloff(y=y, sr=sr)
    mfcc = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chromagram = lib.feature.chroma_stft(y=y, sr=sr) # Harmonic/pitch analysis
    tempo_bt, _ = lib.beat.beat_track(y=y, sr=sr, units='time')

    ### Data Normalization ###

    # Open input
    with open(input_json_path, mode="r") as f:
        input_json_data = json.load(f) # Type array

    # Open output
    with open(output_json_path, mode="r") as f:
        output_json_data = json.load(f)
    
    # Declare empty json object to append to overall array later
    json_object = copy.deepcopy(input_json_data[json_index])

    print("Here is the json_object:", json_object)

    # Tempo and Tempo_bt
    json_object["tempo"] = float(tempo[0])
    json_object["tempo_bt"] = float(tempo_bt[0])

    # RMS
    assign_summary(json_object, rms, "rms")

    # ZCR
    assign_summary(json_object, zcr, "zcr")

    # Spec Centroid
    assign_summary(json_object, spec_centroid, "spec_centroid")

    # Spec Bandwidth
    assign_summary(json_object, spec_bandwidth, "spec_bandwidth")

    # Spec Contrast
    assign_summary(json_object, spec_contrast, "spec_contrast")

    # Spec Flatness
    assign_summary(json_object, spec_flatness, "spec_flatness")

    # Spec Rolloff
    assign_summary(json_object, spec_rolloff, "spec_rolloff")

    # MFCC
    assign_summary(json_object, mfcc, "mfcc")

    # Chromagram
    assign_summary(json_object, chromagram, "chromagram")

    # Append the completed json_object into the json_data array
    output_json_data.append(json_object)

    # Dump the new addition the .json file featured in parameters
    with open(output_json_path, mode="w") as f:
        json.dump(output_json_data, f, indent=2)
    
    print("Done!")


input_json_path = r"server\sentiment_model\muse_v3.json"
output_json_path = r"server\sentiment_model\output.json"

# # Test function runs:
# analyze_audio(audio_path, input_json_path, output_json_path, json_index=0)
# analyze_audio(audio_path, input_json_path, output_json_path, json_index=1)

# ----------------------------------------- MAIN -------------------------------------------------
# Create a state-changing function to communicate between .py and .tsx
# state_path = r"server/sentiment_model/state.json"

# def write_status(status):
#     with open(state_path, "w") as f:
#         json.dump({"state": status}, f)

# MUSE DATASET READ
with open(input_json_path, mode="r") as f:
    input_json_data = json.load(f)

data_length = len(input_json_data)

current_track_path = r"server\sentiment_model\current_index.json"

# CURRENT INDEX READ
def read_current_index():
    with open(current_track_path, mode="r") as f:
        current_index_json = json.load(f)
    
    return current_index_json

def write_index(index, status):
    with open(current_track_path, mode="w") as f:
        json.dump({"status": status,
                   "index": index, 
                   "spotify_id": input_json_data[index]["spotify_id"]
                   }, f, indent=2)

# MUST HAVE FIRST SONG IN DATASET ALREADY IN CURRENT_INDEX.JSON

audio_path = r"server\sentiment_model\output.wav"


def process_song(index):
    # 1. Open selenium and recprd song
    open_selenium()
    print(f"Processing song {index}: Selenium opened")

    # 2. Extract and record audio features
    analyze_audio(audio_path, input_json_path, output_json_path, index)
    print(f"Song {index}: Audio analyzed")

    # 3. Update output .json file for next song
    write_index(index + 1, 1)
    print(f"Song {index}: Updated index to {index + 1}")

def main(start_index):
    index = start_index
    
    while(index < data_length):
        try:
            print(f"\n=== Processing song {index}/{data_length} ===")
            process_song(index)
            index += 1

        except Exception as e:
            print(f"Error processing song {index}: {e}")
            write_index(index, 0)
            break
    
    print("Done!")

# def main(index): # Should have index to begin at just in case process gets paused
#     # 1. Open selenium and record song
#     open_selenium()

#     print("1. Passed!")

#     # 2. Extract and record audio features
#     analyze_audio(audio_path, input_json_path, output_json_path, index)

#     # If there exists a next entry, grab the spotify_id of that entry
#     if index + 1 <= data_length:
#         # 3. Update output .json file
#         write_index(index + 1, 1)

#         print("2. Passed!")
#     else:
#         print("Done!")
#         return

if __name__ == "__main__":
    current_state = read_current_index()
    start = current_state["index"]
    print(f"Starting from index {start}")

    main(start_index=start)

# TODO: FIGURE OUT THIS FILE I/O SYSTEM