import csv
import pandas as pd

original_path = r"server\sentiment_model\muse_v3.csv"
modified_path = r"server\sentiment_model\muse_v3_modified.csv"


df = pd.read_csv(original_path)

# Tempo
df["tempo"] = None
df["tempo_bt"] = None

# RMS
df["rms_mean"] = None
df["rms_std"] = None
df["rms_min"] = None
df["rms_max"] = None

# ZCR
df["zcr_mean"] = None
df["zcr_std"] = None
df["zcr_min"] = None
df["zcr_max"] = None

# Centroid
df["spec_centroid_mean"] = None
df["spec_centroid_std"] = None
df["spec_centroid_min"] = None
df["spec_centroid_max"] = None

# Bandwidth
df["spec_bandwidth_mean"] = None
df["spec_bandwidth_std"] = None
df["spec_bandwidth_min"] = None
df["spec_bandwidth_max"] = None

# Contrast (7 bands)
df["spec_contrast_1_mean"] = None
df["spec_contrast_1_std"] = None
df["spec_contrast_1_min"] = None
df["spec_contrast_1_max"] = None

df["spec_contrast_2_mean"] = None
df["spec_contrast_2_std"] = None
df["spec_contrast_2_min"] = None
df["spec_contrast_2_max"] = None

df["spec_contrast_3_mean"] = None
df["spec_contrast_3_std"] = None
df["spec_contrast_3_min"] = None
df["spec_contrast_3_max"] = None

df["spec_contrast_4_mean"] = None
df["spec_contrast_4_std"] = None
df["spec_contrast_4_min"] = None
df["spec_contrast_4_max"] = None

df["spec_contrast_5_mean"] = None
df["spec_contrast_5_std"] = None
df["spec_contrast_5_min"] = None
df["spec_contrast_5_max"] = None

df["spec_contrast_6_mean"] = None
df["spec_contrast_6_std"] = None
df["spec_contrast_6_min"] = None
df["spec_contrast_6_max"] = None

df["spec_contrast_7_mean"] = None
df["spec_contrast_7_std"] = None
df["spec_contrast_7_min"] = None
df["spec_contrast_7_max"] = None

# Rolloff
df["spec_rolloff_mean"] = None
df["spec_rolloff_std"] = None
df["spec_rolloff_min"] = None
df["spec_rolloff_max"] = None

# Flatness
df["spec_flatness_mean"] = None
df["spec_flatness_std"] = None
df["spec_flatness_min"] = None
df["spec_flatness_max"] = None

# MFCC (13 bands)
df["mfcc_1_mean"] = None
df["mfcc_1_std"] = None
df["mfcc_1_min"] = None
df["mfcc_1_max"] = None

df["mfcc_2_mean"] = None
df["mfcc_2_std"] = None
df["mfcc_2_min"] = None
df["mfcc_2_max"] = None

df["mfcc_3_mean"] = None
df["mfcc_3_std"] = None
df["mfcc_3_min"] = None
df["mfcc_3_max"] = None

df["mfcc_4_mean"] = None
df["mfcc_4_std"] = None
df["mfcc_4_min"] = None
df["mfcc_4_max"] = None

df["mfcc_5_mean"] = None
df["mfcc_5_std"] = None
df["mfcc_5_min"] = None
df["mfcc_5_max"] = None

df["mfcc_6_mean"] = None
df["mfcc_6_std"] = None
df["mfcc_6_min"] = None
df["mfcc_6_max"] = None

df["mfcc_7_mean"] = None
df["mfcc_7_std"] = None
df["mfcc_7_min"] = None
df["mfcc_7_max"] = None

df["mfcc_8_mean"] = None
df["mfcc_8_std"] = None
df["mfcc_8_min"] = None
df["mfcc_8_max"] = None

df["mfcc_9_mean"] = None
df["mfcc_9_std"] = None
df["mfcc_9_min"] = None
df["mfcc_9_max"] = None

df["mfcc_10_mean"] = None
df["mfcc_10_std"] = None
df["mfcc_10_min"] = None
df["mfcc_10_max"] = None

df["mfcc_11_mean"] = None
df["mfcc_11_std"] = None
df["mfcc_11_min"] = None
df["mfcc_11_max"] = None

df["mfcc_12_mean"] = None
df["mfcc_12_std"] = None
df["mfcc_12_min"] = None
df["mfcc_12_max"] = None

df["mfcc_13_mean"] = None
df["mfcc_13_std"] = None
df["mfcc_13_min"] = None
df["mfcc_13_max"] = None

# Chromagram (12 bands)
df["chromagram_1_mean"] = None
df["chromagram_1_std"] = None
df["chromagram_1_min"] = None
df["chromagram_1_max"] = None

df["chromagram_2_mean"] = None
df["chromagram_2_std"] = None
df["chromagram_2_min"] = None
df["chromagram_2_max"] = None

df["chromagram_3_mean"] = None
df["chromagram_3_std"] = None
df["chromagram_3_min"] = None
df["chromagram_3_max"] = None

df["chromagram_4_mean"] = None
df["chromagram_4_std"] = None
df["chromagram_4_min"] = None
df["chromagram_4_max"] = None

df["chromagram_5_mean"] = None
df["chromagram_5_std"] = None
df["chromagram_5_min"] = None
df["chromagram_5_max"] = None

df["chromagram_6_mean"] = None
df["chromagram_6_std"] = None
df["chromagram_6_min"] = None
df["chromagram_6_max"] = None

df["chromagram_7_mean"] = None
df["chromagram_7_std"] = None
df["chromagram_7_min"] = None
df["chromagram_7_max"] = None

df["chromagram_8_mean"] = None
df["chromagram_8_std"] = None
df["chromagram_8_min"] = None
df["chromagram_8_max"] = None

df["chromagram_9_mean"] = None
df["chromagram_9_std"] = None
df["chromagram_9_min"] = None
df["chromagram_9_max"] = None

df["chromagram_10_mean"] = None
df["chromagram_10_std"] = None
df["chromagram_10_min"] = None
df["chromagram_10_max"] = None

df["chromagram_11_mean"] = None
df["chromagram_11_std"] = None
df["chromagram_11_min"] = None
df["chromagram_11_max"] = None

df["chromagram_12_mean"] = None
df["chromagram_12_std"] = None
df["chromagram_12_min"] = None
df["chromagram_12_max"] = None

columns = df.columns.to_list()
print(columns)