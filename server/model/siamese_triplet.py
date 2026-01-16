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

from scipy.spatial.distance import jaccard

# raw = """acerbic,aggressive,agreeable,airy,ambitious,amiable/good-natured,angry,angst-ridden,anguished/distraught,angular,animated,anthemic,apocalyptic,arid,athletic,atmospheric,austere,autumnal,belligerent,benevolent,bitter,bittersweet,bleak,boisterous,bombastic,bouncy,brash,brassy,bravado,bright,brittle,brooding,calm/peaceful,campy,capricious,carefree,cartoonish,cathartic,celebratory,cerebral,cheerful,child-like,circular,clinical,cold,comic,complex,concise,confessional,confident,confrontational,cosmopolitan,crunchy,cute,cynical/sarcastic,dark,declamatory,defiant,delicate,demonic,desperate,detached,devotional,difficult,dignified/noble,dissonant,dramatic,dreamy,driving,druggy,earnest,earthy,ebullient,eccentric,ecstatic,eerie,effervescent,elaborate,elegant,elegiac,energetic,enigmatic,epic,erotic,ethereal,euphoric,exciting,exotic,exploratory,explosive,extroverted,exuberant,fantastic/fantasy-like,feral,feverish,fierce,fiery,flashy,flowing,fractured,freewheeling,fun,funereal,gentle,giddy,gleeful,gloomy,graceful,greasy,grim,gritty,gutsy,happy.1,harsh,heavy,hedonistic,heroic,hostile,humorous,hungry,hymn-like,hyper,hypnotic,improvisatory,indulgent,innocent,insular,intense,intimate,introspective,ironic,irreverent,jovial,joyous,kinetic,knotty,laid-back/mellow,languid,lazy,light,literate,lively,lonely,loose,lush,lyrical,macabre,magical,majestic,malevolent,manic,marching,martial,meandering,mechanical,meditative,melancholy,melodic,menacing,messy,mighty,monastic,monumental,motoric,mysterious,mystical,naive,narcotic,narrative,negative,nervous/jittery,nihilistic,nocturnal,nostalgic,ominous,optimistic,opulent,organic,ornate,outraged,outrageous,paranoid,passionate,pastoral,patriotic,perky,philosophical,plain,plaintive,playful,poetic,poignant,positive,powerful,precious,provocative,pulsing,pure,quaint,quirky,radiant,rambunctious,ramshackle,raucous,reassuring/consoling,rebellious,reckless,refined,reflective,regretful,relaxed,reserved,resolute,restrained,reverent,rhapsodic,rollicking,romantic,rousing,rowdy,rustic,sacred,sad.1,sarcastic,sardonic,satirical,savage,scary,scattered,searching,seductive,self-conscious,sensual,sentimental,serious,severe,sexual,sexy,shimmering,silly,sleazy,slick,smooth,snide,soft/quiet,somber,soothing,sophisticated,spacey,spacious,sparkling,sparse,spicy,spiritual,spontaneous,spooky,sprawling,sprightly,springlike,stately,street-smart,striding,strong,stylish,suffocating,sugary,summery,sunny,suspenseful,swaggering,sweet,swinging,technical,tender,tense/anxious,theatrical,thoughtful,threatening,thrilling,tight,tough,tragic,transparent/translucent,trashy,trippy,triumphant,turbulent,uncompromising,understated,unsettling,uplifting,urgent,virile,visceral,volatile,vulgar,vulnerable,warm,weary,whimsical,wintry,wistful,witty,wry,yearning"""

# seeds = [s.strip() for s in raw.split(",")] # Strip and split raw text into array of quoted elements

seed_labels = ['acerbic', 'aggressive', 'agreeable', 'airy', 'ambitious', 'amiable/good-natured', 'angry', 'angst-ridden', 'anguished/distraught', 'angular', 'animated', 'anthemic', 'apocalyptic', 'arid', 'athletic', 'atmospheric', 'austere', 'autumnal', 'belligerent', 'benevolent', 'bitter', 'bittersweet', 'bleak', 'boisterous', 'bombastic', 'bouncy', 'brash', 'brassy', 'bravado', 'bright', 'brittle', 'brooding', 'calm/peaceful', 'campy', 'capricious', 'carefree', 'cartoonish', 'cathartic', 'celebratory', 'cerebral', 'cheerful', 'child-like', 'circular', 'clinical', 'cold', 'comic', 'complex', 'concise', 'confessional', 'confident', 'confrontational', 'cosmopolitan', 'crunchy', 'cute', 'cynical/sarcastic', 'dark', 'declamatory', 'defiant', 'delicate', 'demonic', 'desperate', 'detached', 'devotional', 'difficult', 'dignified/noble', 'dissonant', 'dramatic', 'dreamy', 'driving', 'druggy', 'earnest', 'earthy', 'ebullient', 'eccentric', 'ecstatic', 'eerie', 'effervescent', 'elaborate', 'elegant', 'elegiac', 'energetic', 'enigmatic', 'epic', 'erotic', 'ethereal', 'euphoric', 'exciting', 'exotic', 'exploratory', 'explosive', 'extroverted', 'exuberant', 'fantastic/fantasy-like', 'feral', 'feverish', 'fierce', 'fiery', 'flashy', 'flowing', 'fractured', 'freewheeling', 'fun', 'funereal', 'gentle', 'giddy', 'gleeful', 'gloomy', 'graceful', 'greasy', 'grim', 'gritty', 'gutsy', 'happy.1', 'harsh', 'heavy', 'hedonistic', 'heroic', 'hostile', 'humorous', 'hungry', 'hymn-like', 'hyper', 'hypnotic', 'improvisatory', 'indulgent', 'innocent', 'insular', 'intense', 'intimate', 'introspective', 'ironic', 'irreverent', 'jovial', 'joyous', 'kinetic', 'knotty', 'laid-back/mellow', 'languid', 'lazy', 'light', 'literate', 'lively', 'lonely', 'loose', 'lush', 'lyrical', 'macabre', 'magical', 'majestic', 'malevolent', 'manic', 'marching', 'martial', 'meandering', 'mechanical', 'meditative', 'melancholy', 'melodic', 'menacing', 'messy', 'mighty', 'monastic', 'monumental', 'motoric', 'mysterious', 'mystical', 'naive', 'narcotic', 'narrative', 'negative', 'nervous/jittery', 'nihilistic', 'nocturnal', 'nostalgic', 'ominous', 'optimistic', 'opulent', 'organic', 'ornate', 'outraged', 'outrageous', 'paranoid', 'passionate', 'pastoral', 'patriotic', 'perky', 'philosophical', 'plain', 'plaintive', 'playful', 'poetic', 'poignant', 'positive', 'powerful', 'precious', 'provocative', 'pulsing', 'pure', 'quaint', 'quirky', 'radiant', 'rambunctious', 'ramshackle', 'raucous', 'reassuring/consoling', 'rebellious', 'reckless', 'refined', 'reflective', 'regretful', 'relaxed', 'reserved', 'resolute', 'restrained', 'reverent', 'rhapsodic', 'rollicking', 'romantic', 'rousing', 'rowdy', 'rustic', 'sacred', 'sad.1', 'sarcastic', 'sardonic', 'satirical', 'savage', 'scary', 'scattered', 'searching', 'seductive', 'self-conscious', 'sensual', 'sentimental', 'serious', 'severe', 'sexual', 'sexy', 'shimmering', 'silly', 'sleazy', 'slick', 'smooth', 'snide', 'soft/quiet', 'somber', 'soothing', 'sophisticated', 'spacey', 'spacious', 'sparkling', 'sparse', 'spicy', 'spiritual', 'spontaneous', 'spooky', 'sprawling', 'sprightly', 'springlike', 'stately', 'street-smart', 'striding', 'strong', 'stylish', 'suffocating', 'sugary', 'summery', 'sunny', 'suspenseful', 'swaggering', 'sweet', 'swinging', 'technical', 'tender', 'tense/anxious', 'theatrical', 'thoughtful', 'threatening', 'thrilling', 'tight', 'tough', 'tragic', 'transparent/translucent', 'trashy', 'trippy', 'triumphant', 'turbulent', 'uncompromising', 'understated', 'unsettling', 'uplifting', 'urgent', 'virile', 'visceral', 'volatile', 'vulgar', 'vulnerable', 'warm', 'weary', 'whimsical', 'wintry', 'wistful', 'witty', 'wry', 'yearning']

seed_classification = {
                        "positive": [
                            "agreeable",
                            "airy",
                            "amiable/good-natured",
                            "animated",
                            "bouncy",
                            "bright",
                            "carefree",
                            "cheerful",
                            "child-like",
                            "cute",
                            "ecstatic",
                            "effervescent",
                            "energetic",
                            "enthusiastic",
                            "exuberant",
                            "exciting",
                            "fun",
                            "giddy",
                            "gleeful",
                            "happy",
                            "hyper",
                            "innocent",
                            "jovial",
                            "joyous",
                            "kinetic",
                            "laid-back/mellow",
                            "light",
                            "lively",
                            "loose",
                            "naive",
                            "optimistic",
                            "perky",
                            "playful",
                            "positive",
                            "radiant",
                            "rousing",
                            "sugary",
                            "summery",
                            "sunny",
                            "sweet",
                            "uplifting",
                            "warm",
                            "sprightly",
                            "witty",
                            "flashy",
                            "shimmering",
                            "silly",
                            "loungy",
                            "exotic",
                            "funereal",
                            "outrageous",
                            "humorous",
                            "gutsy",
                            "street-smart",
                            "swaggering",
                            "rollicking",
                            "bouncy"
                        ],
                        "negative": [
                            "acerbic",
                            "angry",
                            "angst-ridden",
                            "anguished/distraught",
                            "arid",
                            "bitter",
                            "bleak",
                            "brittle",
                            "brooding",
                            "cold",
                            "cynical/sarcastic",
                            "dark",
                            "desperate",
                            "dissonant",
                            "elegiac",
                            "gloomy",
                            "greasy",
                            "grim",
                            "gritty",
                            "harsh",
                            "hostile",
                            "nihilistic",
                            "nocturnal",
                            "regretful",
                            "sad",
                            "sarcastic",
                            "sardonic",
                            "satirical",
                            "scary",
                            "severe",
                            "sleezy",
                            "trashy",
                            "turbulent",
                            "tense/anxious",
                            "unsettling",
                            "vulgar",
                            "weary",
                            "wistful",
                            "vulnerable",
                            "paranoid",
                            "menacing",
                            "feral",
                            "ominous",
                            "funereal",
                            "desperate",
                            "plaintive",
                            "poignant",
                            "self-conscious"
                        ],
                        "aggressive": [
                            "aggressive",
                            "belligerent",
                            "confrontational",
                            "defiant",
                            "driving",
                            "explosive",
                            "fiery",
                            "gutsy",
                            "intense",
                            "manic",
                            "outraged",
                            "powerful",
                            "reckless",
                            "rebellious",
                            "rowdy",
                            "savage",
                            "threatening",
                            "tough",
                            "urgent",
                            "virile",
                            "visceral",
                            "volatile"
                        ],
                        "calm": [
                            "airy",
                            "atmospheric",
                            "austere",
                            "autumnal",
                            "brooding",
                            "calm/peaceful",
                            "delicate",
                            "detached",
                            "dreamy",
                            "ethereal",
                            "hypnotic",
                            "introspective",
                            "laid-back/mellow",
                            "languid",
                            "light",
                            "meandering",
                            "meditative",
                            "melodic",
                            "plain",
                            "relaxed",
                            "reserved",
                            "restrained",
                            "soothing",
                            "smooth/quiet",
                            "spacious",
                            "tender",
                            "thoughtful",
                            "wintry"
                        ],
                        "romantic": [
                            "bittersweet",
                            "cathartic",
                            "confessional",
                            "delicate",
                            "desperate",
                            "earnest",
                            "flowing",
                            "intimate",
                            "introspective",
                            "lyrical",
                            "melancholy",
                            "melodic",
                            "narrative",
                            "nostalgic",
                            "poetic",
                            "poignant",
                            "rhapsodic",
                            "romantic",
                            "seductive",
                            "sensual",
                            "sentimental",
                            "sweet",
                            "tender",
                            "vulnerable",
                            "wistful"
                        ],
                        "experimental": [
                            "angular",
                            "cerebral",
                            "complex",
                            "crunchy",
                            "capricious",
                            "circular",
                            "clinical",
                            "concise",
                            "cynical/sarcastic",
                            "detached",
                            "druggy",
                            "eccentric",
                            "elaborate",
                            "enigmatic",
                            "ethereal",
                            "exotic",
                            "exploratory",
                            "fractured",
                            "freewheeling",
                            "improvisatory",
                            "indulgent",
                            "insular",
                            "knotty",
                            "literate",
                            "messy",
                            "mechanical",
                            "mysterious",
                            "mystical",
                            "narcotic",
                            "narrative",
                            "ornate",
                            "philosophical",
                            "quirky",
                            "quaint",
                            "scattered",
                            "searching",
                            "sexual",
                            "spacey",
                            "sparse",
                            "spontaneous",
                            "stylish",
                            "technical",
                            "theatrical",
                            "trippy",
                            "sprawling"
                        ],
                        "epic": [
                            "ambitious",
                            "anthemic",
                            "apocalyptic",
                            "brassy",
                            "bravado",
                            "bombastic",
                            "celebratory",
                            "confident",
                            "declaratory",
                            "dramatic",
                            "elaborate",
                            "epic",
                            "fiery",
                            "flashy",
                            "heroic",
                            "majestic",
                            "mighty",
                            "monumental",
                            "opulent",
                            "passionate",
                            "patriotic",
                            "rhapsodic",
                            "resolute",
                            "restrained",
                            "stately",
                            "striding",
                            "strong",
                            "triumphant",
                            "urgent",
                            "virile",
                            "visceral",
                            "shimmering",
                            "rousing",
                            "exciting",
                            "thrilling"
                        ],
                        "humorous": [
                            "campy",
                            "cartoonish",
                            "comic",
                            "ironic",
                            "irreverent",
                            "outrageous",
                            "playful",
                            "sarcastic",
                            "sardonic",
                            "satirical",
                            "silly",
                            "trashy",
                            "witty"
                        ],
                        "spiritual": [
                            "devotional",
                            "hymn-like",
                            "meditative",
                            "monastic",
                            "mystical",
                            "reverent",
                            "sacred"
                        ],
                        "kinetic": [
                            "athletic",
                            "animated",
                            "driving",
                            "hyper",
                            "lively",
                            "marching",
                            "motoric",
                            "pulsing",
                            "rambunctious",
                            "rowdy",
                            "striding",
                            "swinging",
                            "thrilling"
                        ],
                        "gothic_dark": [
                            "dark",
                            "demonic",
                            "eerie",
                            "funereal",
                            "macabre",
                            "malevolent",
                            "nihilistic",
                            "ominous",
                            "paranoid",
                            "scary",
                            "spooky",
                            "suffocating",
                            "suspenseful"
                        ],
                        "folk": ["earthy", "organic", "pastoral", "rustic"]
                    }

seed_to_class = {}

# Invert classes to seeds -> seed to class
for cls, seeds in seed_classes.items():
    for seed in seeds:
        seed_to_class[seed] = cls

# raw_genres = """rap,metal,hip-hop,nu metal,singer-songwriter,punk,industrial,metalcore,alternative metal,classic rock,electroclash,rock,post-hardcore,progressive metal,indie rock,indie pop,indie,pop,post-punk,thrash metal,industrial metal,gothic metal,death metal,alternative rock,screamo,noise rock,electronic,riot grrrl,electro,symphonic metal,grunge,trip-hop,hard rock,breakbeat,melodic death metal,hip hop,hardcore,alternative,experimental,country,stoner rock,horrorcore,ska,black metal,dark electro,redneck,hardcore punk,stoner metal,ambient,underground hip hop,math rock,grindcore,doom metal,noise pop,emo,deathcore,ebm,post-metal,goth,bluegrass,british,christian metal,soul,industrial rock,folk,digital hardcore,post-rock,visual kei,dance,britpop,german,mathcore,blues rock,melodic hardcore,minimal techno,new wave,avant-garde,pop punk,groove metal,christian rock,j-rock,funk,breakcore,folk metal,crust punk,funk metal,acoustic,oi,soundtrack,idm,melodic black metal,drum and bass,anime,electro-industrial,blues,k-pop,d-beat,aggrotech,dubstep,rockabilly,house,power metal,art pop,progressive rock,noise,gothic rock,sludge metal,psychobilly,electronic rock,spanish,piano rock,piano,dark cabaret,experimental rock,horror punk,lo-fi,dance rock,folk punk,summer,comedy,jazz,depressive black metal,poetry,cybergrind,pop rock,powerviolence,sad,finnish metal,classical,electronica,grime,shoegaze,crunk,guitar,rock en espanol,slowcore,violin,spoken word,neofolk,psychedelic rock,canadian rock,dream pop,technical death metal,madchester,garage rock,drone,progressive black metal,downtempo,french,synthwave,worship,contemporary classical,zeuhl,vocal trance,mpb,latin,trance,techno,chill,cabaret,ambient pop,breaks,r&b,world,krautrock,reggae,boogie,soft rock,washboard,video game music,disco,minimal synth,dark ambient,art rock,medieval,darkstep,erhu,hardstyle,j-pop,groove,garage,martial industrial,russian alternative,minimalism,new weird america,symphonic rock,atmospheric black metal,jungle,neoclassical darkwave,deathrock,psychedelic pop,experimental folk,trip hop,progressive house,ritual ambient,celtic,free jazz,water,cyberpunk,halloween,soundtracks,orchestra,skate punk,teen pop,afrobeat,new age,electropop,dancehall,quran,atmosphere,a cappella,southern rock,happy,french rock,funk carioca,electro house,world fusion,garage punk,underground rap,choral,folktronica,latin rock,brazil,symphonic death metal,glitch hop,drone metal,synthpop,hard trance,glam rock,meme rap,bossa nova,witch house,sleep,dark jazz,post-black metal,trap,dark wave,cello,harp,hip house,hauntology,easy listening,acid jazz,futurepop,western swing,melodic rap,ambient black metal,glitch,music box,avant-garde jazz,indietronica,accordion,lullaby,deep ambient,disney,psychedelic trance,alternative country,folk rock,lounge,beats,alternative pop,dub,abstract,minimal wave,space rock,nu jazz,uk garage,coldwave,turntablism,lilith,power electronics,swing,northern soul,kids,novelty,steampunk,calypso,russian rock,goregrind,gypsy,ccm,neurofunk,tango,west coast rap,celtic rock,chanson,alternative dance,8-bit,french pop,nederpop,outsider,swedish,chamber pop,vocal jazz,ukulele,indie folk,chillwave,praise,songwriter,tex-mex,native american,meditation,smooth jazz,psychill,organic ambient,big beat,opera,zen,noisecore,djent,bachata,freak folk,balearic,ska punk,romance,power pop,houston rap,vaporwave,post-grunge,rock nacional,chillstep,ambient industrial,dark folk,jazz fusion,mandolin,jangle pop,broken beat,klezmer,street punk,doo-wop,motown,merseybeat,straight edge,gospel,chiptune,polka,jam band,minecraft,gypsy punk,parody,suomisaundi,gypsy jazz,rhythm and blues,samba,merengue,baroque,taiko,spanish classical,hawaiian,flamenco,rumba,fado,schlager,turkish,latin jazz,fingerstyle,qawwali,broadway,focus,rave,crack rock steady,big band,contemporary jazz,party,freestyle,latin pop,progressive breaks,eurodance,no wave,brass band,modern rock,club,polish rock,celtic punk,soca,boogaloo,neo soul,electro jazz,psychedelic folk,indian,garage rock revival,tropicalia,jazz funk,anti-folk,future garage,demoscene,goa trance,progressive trance,moog,deep house,space age pop,tech house,yoga,deep techno,sufi,dark disco,new jack swing"""

genre_labels = ['rap', 'metal', 'hip-hop', 'nu metal', 'singer-songwriter', 'punk', 'industrial', 'metalcore', 'alternative metal', 'classic rock', 'electroclash', 'rock', 'post-hardcore', 'progressive metal', 'indie rock', 'indie pop', 'indie', 'pop', 'post-punk', 'thrash metal', 'industrial metal', 'gothic metal', 'death metal', 'alternative rock', 'screamo', 'noise rock', 'electronic', 'riot grrrl', 'electro', 'symphonic metal', 'grunge', 'trip-hop', 'hard rock', 'breakbeat', 'melodic death metal', 'hip hop', 'hardcore', 'alternative', 'experimental', 'country', 'stoner rock', 'horrorcore', 'ska', 'black metal', 'dark electro', 'redneck', 'hardcore punk', 'stoner metal', 'ambient', 'underground hip hop', 'math rock', 'grindcore', 'doom metal', 'noise pop', 'emo', 'deathcore', 'ebm', 'post-metal', 'goth', 'bluegrass', 'british', 'christian metal', 'soul', 'industrial rock', 'folk', 'digital hardcore', 'post-rock', 'visual kei', 'dance', 'britpop', 'german', 'mathcore', 'blues rock', 'melodic hardcore', 'minimal techno', 'new wave', 'avant-garde', 'pop punk', 'groove metal', 'christian rock', 'j-rock', 'funk', 'breakcore', 'folk metal', 'crust punk', 'funk metal', 'acoustic', 'oi', 'soundtrack', 'idm', 'melodic black metal', 'drum and bass', 'anime', 'electro-industrial', 'blues', 'k-pop', 'd-beat', 'aggrotech', 'dubstep', 'rockabilly', 'house', 'power metal', 'art pop', 'progressive rock', 'noise', 'gothic rock', 'sludge metal', 'psychobilly', 'electronic rock', 'spanish', 'piano rock', 'piano', 'dark cabaret', 'experimental rock', 'horror punk', 'lo-fi', 'dance rock', 'folk punk', 'summer', 'comedy', 'jazz', 'depressive black metal', 'poetry', 'cybergrind', 'pop rock', 'powerviolence', 'sad', 'finnish metal', 'classical', 'electronica', 'grime', 'shoegaze', 'crunk', 'guitar', 'rock en espanol', 'slowcore', 'violin', 'spoken word', 'neofolk', 'psychedelic rock', 'canadian rock', 'dream pop', 'technical death metal', 'madchester', 'garage rock', 'drone', 'progressive black metal', 'downtempo', 'french', 'synthwave', 'worship', 'contemporary classical', 'zeuhl', 'vocal trance', 'mpb', 'latin', 'trance', 'techno', 'chill', 'cabaret', 'ambient pop', 'breaks', 'r&b', 'world', 'krautrock', 'reggae', 'boogie', 'soft rock', 'washboard', 'video game music', 'disco', 'minimal synth', 'dark ambient', 'art rock', 'medieval', 'darkstep', 'erhu', 'hardstyle', 'j-pop', 'groove', 'garage', 'martial industrial', 'russian alternative', 'minimalism', 'new weird america', 'symphonic rock', 'atmospheric black metal', 'jungle', 'neoclassical darkwave', 'deathrock', 'psychedelic pop', 'experimental folk', 'trip hop', 'progressive house', 'ritual ambient', 'celtic', 'free jazz', 'water', 'cyberpunk', 'halloween', 'soundtracks', 'orchestra', 'skate punk', 'teen pop', 'afrobeat', 'new age', 'electropop', 'dancehall', 'quran', 'atmosphere', 'a cappella', 'southern rock', 'happy', 'french rock', 'funk carioca', 'electro house', 'world fusion', 'garage punk', 'underground rap', 'choral', 'folktronica', 'latin rock', 'brazil', 'symphonic death metal', 'glitch hop', 'drone metal', 'synthpop', 'hard trance', 'glam rock', 'meme rap', 'bossa nova', 'witch house', 'sleep', 'dark jazz', 'post-black metal', 'trap', 'dark wave', 'cello', 'harp', 'hip house', 'hauntology', 'easy listening', 'acid jazz', 'futurepop', 'western swing', 'melodic rap', 'ambient black metal', 'glitch', 'music box', 'avant-garde jazz', 'indietronica', 'accordion', 'lullaby', 'deep ambient', 'disney', 'psychedelic trance', 'alternative country', 'folk rock', 'lounge', 'beats', 'alternative pop', 'dub', 'abstract', 'minimal wave', 'space rock', 'nu jazz', 'uk garage', 'coldwave', 'turntablism', 'lilith', 'power electronics', 'swing', 'northern soul', 'kids', 'novelty', 'steampunk', 'calypso', 'russian rock', 'goregrind', 'gypsy', 'ccm', 'neurofunk', 'tango', 'west coast rap', 'celtic rock', 'chanson', 'alternative dance', '8-bit', 'french pop', 'nederpop', 'outsider', 'swedish', 'chamber pop', 'vocal jazz', 'ukulele', 'indie folk', 'chillwave', 'praise', 'songwriter', 'tex-mex', 'native american', 'meditation', 'smooth jazz', 'psychill', 'organic ambient', 'big beat', 'opera', 'zen', 'noisecore', 'djent', 'bachata', 'freak folk', 'balearic', 'ska punk', 'romance', 'power pop', 'houston rap', 'vaporwave', 'post-grunge', 'rock nacional', 'chillstep', 'ambient industrial', 'dark folk', 'jazz fusion', 'mandolin', 'jangle pop', 'broken beat', 'klezmer', 'street punk', 'doo-wop', 'motown', 'merseybeat', 'straight edge', 'gospel', 'chiptune', 'polka', 'jam band', 'minecraft', 'gypsy punk', 'parody', 'suomisaundi', 'gypsy jazz', 'rhythm and blues', 'samba', 'merengue', 'baroque', 'taiko', 'spanish classical', 'hawaiian', 'flamenco', 'rumba', 'fado', 'schlager', 'turkish', 'latin jazz', 'fingerstyle', 'qawwali', 'broadway', 'focus', 'rave', 'crack rock steady', 'big band', 'contemporary jazz', 'party', 'freestyle', 'latin pop', 'progressive breaks', 'eurodance', 'no wave', 'brass band', 'modern rock', 'club', 'polish rock', 'celtic punk', 'soca', 'boogaloo', 'neo soul', 'electro jazz', 'psychedelic folk', 'indian', 'garage rock revival', 'tropicalia', 'jazz funk', 'anti-folk', 'future garage', 'demoscene', 'goa trance', 'progressive trance', 'moog', 'deep house', 'space age pop', 'tech house', 'yoga', 'deep techno', 'sufi', 'dark disco', 'new jack swing']

genre_classification = {
                        "metal": [
                            "metal",
                            "nu metal",
                            "metalcore",
                            "alternative metal",
                            "progressive metal",
                            "thrash metal",
                            "industrial metal",
                            "gothic metal",
                            "death metal",
                            "symphonic metal",
                            "melodic death metal",
                            "black metal",
                            "stoner metal",
                            "doom metal",
                            "deathcore",
                            "grindcore",
                            "post-metal",
                            "christian metal",
                            "mathcore",
                            "folk metal",
                            "groove metal",
                            "funk metal",
                            "melodic black metal",
                            "power metal",
                            "sludge metal",
                            "depressive black metal",
                            "technical death metal",
                            "progressive black metal",
                            "atmospheric black metal",
                            "drone metal",
                            "symphonic death metal",
                            "post-black metal",
                            "ambient black metal",
                            "finnish metal",
                            "goregrind",
                            "djent"
                        ],
                        "hip_hop": [
                            "rap",
                            "hip-hop",
                            "trip-hop",
                            "hip hop",
                            "underground hip hop",
                            "horrorcore",
                            "underground rap",
                            "crunk",
                            "meme rap",
                            "trap",
                            "west coast rap",
                            "houston rap",
                            "melodic rap",
                            "hip house"
                        ],
                        "rock": [
                            "classic rock",
                            "rock",
                            "indie rock",
                            "alternative rock",
                            "noise rock",
                            "grunge",
                            "hard rock",
                            "stoner rock",
                            "math rock",
                            "goth",
                            "industrial rock",
                            "post-rock",
                            "blues rock",
                            "christian rock",
                            "progressive rock",
                            "gothic rock",
                            "electronic rock",
                            "piano rock",
                            "experimental rock",
                            "pop rock",
                            "rock en espanol",
                            "garage rock",
                            "psychedelic rock",
                            "soft rock",
                            "art rock",
                            "symphonic rock",
                            "glam rock",
                            "rockabilly",
                            "psychobilly",
                            "dance rock",
                            "southern rock",
                            "slowcore",
                            "canadian rock",
                            "madchester",
                            "french rock",
                            "latin rock",
                            "celtic rock",
                            "modern rock",
                            "garage rock revival",
                            "post-grunge",
                            "polish rock",
                            "space rock"
                        ],
                        "punk": [
                            "punk",
                            "post-punk",
                            "riot grrrl",
                            "post-hardcore",
                            "hardcore",
                            "hardcore punk",
                            "grindcore",
                            "goth",
                            "digital hardcore",
                            "mathcore",
                            "melodic hardcore",
                            "pop punk",
                            "crust punk",
                            "horror punk",
                            "oi",
                            "skate punk",
                            "garage punk",
                            "d-beat",
                            "folk punk",
                            "powerviolence",
                            "street punk",
                            "ska punk",
                            "gypsy punk",
                            "crack rock steady"
                        ],
                        "electronic": [
                            "electronic",
                            "electroclash",
                            "electro",
                            "breakbeat",
                            "dark electro",
                            "ebm",
                            "digital hardcore",
                            "dance",
                            "minimal techno",
                            "new wave",
                            "breakcore",
                            "aggrotech",
                            "dubstep",
                            "electro-industrial",
                            "electronic rock",
                            "electronica",
                            "idm",
                            "techno",
                            "cyberpunk",
                            "electro house",
                            "drum and bass",
                            "downtempo",
                            "synthwave",
                            "vocal trance",
                            "trance",
                            "chill",
                            "progressive house",
                            "hard trance",
                            "electropop",
                            "glitch hop",
                            "synthpop",
                            "psychedelic trance",
                            "big beat",
                            "uk garage",
                            "coldwave",
                            "turntablism",
                            "8-bit",
                            "chiptune",
                            "suomisaundi",
                            "future garage",
                            "goa trance",
                            "progressive trance",
                            "moog",
                            "deep house",
                            "tech house",
                            "deep techno",
                            "darkstep",
                            "minimal synth",
                            "minimal wave",
                            "power electronics",
                            "witch house",
                            "vaporwave",
                            "electro jazz"
                        ],
                        "experimental": [
                            "experimental",
                            "industrial",
                            "noise pop",
                            "breakcore",
                            "electro-industrial",
                            "avant-garde",
                            "experimental rock",
                            "dark cabaret",
                            "idm",
                            "cyberpunk",
                            "noise",
                            "abstract",
                            "outsider",
                            "lilith",
                            "glitch",
                            "noisecore",
                            "martial industrial",
                            "power electronics",
                            "demoscene",
                            "hauntology"
                        ],
                        "ambient_atmospheric": [
                            "ambient",
                            "atmosphere",
                            "atmospheric black metal",
                            "lo-fi",
                            "ambient pop",
                            "chill",
                            "sleep",
                            "psychill",
                            "organic ambient",
                            "deep ambient",
                            "ritual ambient",
                            "dark ambient",
                            "downtempo",
                            "chillstep"
                        ],
                        "cabaret_theatrical": ["cabaret", "dark cabaret", "broadway", "steampunk"],
                        "funk_rnb": [
                            "soul",
                            "blues rock",
                            "funk",
                            "funk metal",
                            "blues",
                            "r&b",
                            "boogie",
                            "funk carioca",
                            "groove",
                            "northern soul",
                            "motown",
                            "rhythm and blues"
                        ],
                        "folk": [
                            "folk",
                            "folk metal",
                            "neofolk",
                            "experimental folk",
                            "folk rock",
                            "dark folk",
                            "freak folk",
                            "tex-mex",
                            "native american",
                            "celtic"
                        ],
                        "jazz": [
                            "jazz",
                            "avant-garde jazz",
                            "free jazz",
                            "dark jazz",
                            "jazz fusion",
                            "acid jazz",
                            "nu jazz",
                            "contemporary jazz",
                            "vocal jazz",
                            "jazz funk"
                        ],
                        "classical": [
                            "classical",
                            "contemporary classical",
                            "neoclassical darkwave",
                            "medieval",
                            "baroque",
                            "spanish classical",
                            "orchestra"
                        ],
                        "cinematic": [
                            "soundtrack",
                            "video game music",
                            "disney",
                            "soundtracks",
                            "orchestra"
                        ],
                        "pop": [
                            "pop",
                            "indie pop",
                            "singer-songwriter",
                            "noise pop",
                            "britpop",
                            "new wave",
                            "pop punk",
                            "art pop",
                            "pop rock",
                            "teen pop",
                            "summer",
                            "comedy",
                            "psychedelic pop",
                            "alternative pop",
                            "freak folk",
                            "jangle pop",
                            "power pop",
                            "latin pop",
                            "chamber pop",
                            "indietronica",
                            "ambient pop",
                            "dream pop",
                            "indie folk",
                            "chillwave",
                            "electropop"
                        ],
                        "acoustic_instrumental": [
                            "singer-songwriter",
                            "songwriter",
                            "acoustic",
                            "piano",
                            "violin",
                            "cello",
                            "harp",
                            "mandolin",
                            "accordion",
                            "guitar",
                            "ukulele",
                            "fingerstyle",
                            "music box",
                            "lullaby"
                        ],
                        "world": [
                            "latin",
                            "mpb",
                            "brazil",
                            "afrobeat",
                            "world",
                            "world fusion",
                            "calypso",
                            "soca",
                            "bossa nova",
                            "samba",
                            "merengue",
                            "indian",
                            "turkish",
                            "hawaiian",
                            "flamenco",
                            "rumba",
                            "fado",
                            "qawwali",
                            "tango",
                            "native american"
                        ],
                        "european": [
                            "french",
                            "russian alternative",
                            "french pop",
                            "swedish",
                            "canadian rock",
                            "british",
                            "german",
                            "polish rock",
                            "nederpop",
                            "spanish",
                            "spanish classical"
                        ],
                        "asian": ["k-pop", "j-pop", "j-rock", "anime", "indian"],
                        "religious_spiritual": [
                            "worship",
                            "quran",
                            "praise",
                            "ccm",
                            "christian rock",
                            "christian metal"
                        ],
                        "children_family": ["minecraft", "kids", "novelty"],
                        "spoken_performance": ["poetry", "spoken word", "comedy", "parody"],
                        "gothic_dark": ["halloween", "dark wave", "lilith", "witch house"],
                        "electronic_dance": [
                            "dancehall",
                            "boogie",
                            "washboard",
                            "disco",
                            "hardstyle",
                            "house",
                            "club",
                            "party",
                            "rave",
                            "eurodance",
                            "space age pop",
                            "balearic",
                            "lounge",
                            "easy listening"
                        ],
                        "mood_emotive": ["sad", "happy", "focus", "zen", "water"]
                    }

genre_to_class = {}

# Invert classes to genres -> genres to class
for cls, genres in seed_classes.items():
    for genre in genres:
        genre_to_class[genre] = cls

# Semantic similarity function to generate semi-hard triplets
def semantic_similarity(song1, song2):
    # Valence
    valence_weight = 0.25
    valence_distance = jaccard(song1.valence_tag, song2.valence_tag)
    valence_total = valence_weight * valence_distance

    # Arousal
    arousal_weight = 0.25
    arousal_distance = jaccard(song1.arrousal_tag, song2.arousal_tag)
    arousal_total = arousal_weight * arousal_distance

    # Dominance
    dominance_weight = 0.25
    dominance_distance = jaccard(song1.dominance_tag, song2.dominance_tag)
    dominance_total = dominance_weight * dominance_distance

    # Seed
    seed_weight = 0.15
    seed_similarity = 0
    
    song1_seeds = []
    song2_seeds = []

    for label in seed_labels:
        if getattr(song1, label) == 1:
            song1_seeds.append(label)
        if getattr(song2, label) == 1:
            song2_seeds.append(label)

    # Check if song2's seed is within song1's seed class:
    song1_seed_classes = {
        seed_to_class[seed]
        for seed in song1_seeds
        if seed in seed_to_class
    }

    song2_seed_classes = {
        seed_to_class[seed]
        for seed in song2_seeds
        if seed in seed_to_class
    }

    seed_class_overlap = song1_seed_classes & song2_seed_classes

    seed_similarity = len(seed_class_overlap) / max(
        len(song1_seed_classes),
        len(song2_seed_classes),
        1
    )
    
    seed_total = seed_weight * seed_similarity

    # Genre
    genre_weight = 0.1
    genre_similarity = 0

    song1_genres = []
    song2_genres = []

    for label in genre_labels:
        if getattr(song1, label) == 1:
            song1_genres.append(label)
        if getattr(song2, label) == 1:
            song2_genres.append(label)
    
    # Check if song2's genre is within song1's genre class:
    song1_genre_classes = {
        genre_to_class[genre]
        for genre in song1_genres
        if genre in genre_to_class
    }

    song2_genre_classes = {
        genre_to_class[genre]
        for genre in song2_genres
        if genre in genre_to_class
    }

    genre_class_overlap = song1_genre_classes & song2_genre_classes

    genre_similarity = len(genre_class_overlap) / max(
        len(song1_genre_classes),
        len(song2_genre_classes),
        1
    )
    
    genre_total = genre_weight * genre_similarity

    semantic_similarity = valence_total + arousal_total + dominance_total + seed_total + genre_total

    return semantic_similarity

def triplet_gen(batch_num):
    return