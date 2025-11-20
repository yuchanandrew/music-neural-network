import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

path = "model\csv\spotify_dataset.csv"

df = pd.read_csv(path)
# print("Head: ", df.head())'

# Turn categorical data into numerical:
track_encoder = LabelEncoder()
artist_encoder = LabelEncoder()
album_encoder = LabelEncoder()

track_encoded = track_encoder.fit_transform(df['track_name'])
artist_encoded = artist_encoder.fit_transform(df['artist_name'])
album_encoded = album_encoder.fit_transform(df['album_name'])

# Add encoded values to dataframe
df['track_encoded'] = track_encoded
df['artist_encoded'] = artist_encoded
df['album_encoded'] = album_encoded

embed_cols = ['track_encoded', 'artist_encoded', 'album_encoded']

y = df['favor'] # Target Values (output)
X = df.drop(['track_name', 'track_id', 'artist_name', 'album_name', 'favor'], axis=1) # Quantifiable metrics (input)
print("Establishing X and y (complete)")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creating training, testing, and validation data splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# Transforming data
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train.drop(embed_cols, axis=1))
X_val_num = scaler.transform(X_val.drop(embed_cols, axis=1))
X_test_num = scaler.transform(X_test.drop(embed_cols, axis=1))

# Converting to torch tensors:
X_train_tensor = torch.tensor(X_train_num, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

track_train = torch.tensor(X_train['track_encoded'].values, dtype=torch.long)
artist_train = torch.tensor(X_train['artist_encoded'].values, dtype=torch.long)
album_train = torch.tensor(X_train['album_encoded'].values, dtype=torch.long)

# Validation tensors:
X_val_tensor = torch.tensor(X_val_num, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

track_val = torch.tensor(X_val['track_encoded'].values, dtype=torch.long)
artist_val = torch.tensor(X_val['artist_encoded'].values, dtype=torch.long)
album_val = torch.tensor(X_val['album_encoded'].values, dtype=torch.long)

print("Converting training (complete)")

# Defining neural network class
class SimpleNN(nn.Module):
    # Class constructor
    def __init__(self, input_size, hidden_size, output_size, n_tracks, n_artists, n_albums, embed_dim):
        super(SimpleNN, self).__init__()
        # Embedding Vectors:
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.album_emb = nn.Embedding(n_albums, embed_dim)

        # Input Layer
        self.fc1 = nn.Linear(input_size + 3 * embed_dim, hidden_size)

        # Activation Layer
        self.relu = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, track_idx, artist_idx, album_idx):
        track_vec = self.track_emb(track_idx)
        artist_vec = self.artist_emb(artist_idx)
        album_vec = self.album_emb(album_idx)
        
        combined = torch.cat([x, track_vec, artist_vec, album_vec], dim=1)

        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
print("Class definition (complete)")
    
# Training!
torch.manual_seed(42)

# Define input size, hidden size, and output size of NN
input_size = X_train_num.shape[1]
hidden_size = 10
output_size = 1

n_tracks = df['track_encoded'].nunique() + 1
n_artists = df['artist_encoded'].nunique() + 1
n_albums = df['album_encoded'].nunique() + 1
embed_dim = 8

print("Right before model run")

model = SimpleNN(input_size, hidden_size, output_size, n_tracks, n_artists, n_albums, embed_dim)

print("Ran model successfully!")

# Defining loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting model training")
# Training model:
num_epochs = 100 # Minimize overfitting

# Early Stopping:
# This is good for more 'supervised learning' through validation set
best_val_loss = float('inf')
patience = 5 # Number of epochs until early stop assessment
trigger_times = 0

for epoch in range(num_epochs):
    model.train()

    # Forward pass:
    outputs = model(X_train_tensor, track_train, artist_train, album_train)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    with torch.no_grad():
        model.eval()
        
        val_outputs = model(X_val_tensor, track_val, artist_val, album_val)
        val_loss = criterion(val_outputs, y_val_tensor)

        train_prediction = (torch.sigmoid(outputs) > 0.5).int()
        val_prediction = (torch.sigmoid(val_outputs) > 0.5).int()

        train_accuracy = (train_prediction.view(-1) == y_train_tensor.int().view(-1)).float().mean()
        val_accuracy = (val_prediction.view(-1) == y_val_tensor.int().view(-1)).float().mean()

    # Print the loss every 10 epochs:
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | Train Acc: {train_accuracy:.3f} | Val Acc: {val_accuracy:.3f}")
    
    # Early stopping:
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        trigger_times += 1
        
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Testing
print("Starting testing...")

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_num, dtype=torch.float32)
    track_test = torch.tensor(X_test['track_encoded'].values, dtype=torch.long)
    artist_test = torch.tensor(X_test['artist_encoded'].values, dtype=torch.long)
    album_test = torch.tensor(X_test['album_encoded'].values, dtype=torch.long)

    outputs = model(X_test_tensor, track_test, artist_test, album_test)
    predicted = torch.sigmoid(outputs) > 0.5 # Convert logits to 0 or 1
    accuracy = (predicted.int().view(-1) == torch.tensor(y_test.values).int()).sum().item() / len(y_test)

    print(f'Accuracy on the test set: {accuracy:.2f}')

# Loading in custom dataset into dataframe
print("Loading in custom dataset...")
custom_df = pd.read_csv("model\csv\custom_spotify_dataset.csv")

# Turn categorical data into numerical:
print("Transforming custom dataset's track, artist, album into numerical data...")
custom_track_encoded = np.array([
    track_encoder.transform([t])[0] if t in track_encoder.classes_ else n_tracks - 1
    for t in custom_df['track_name']
])

custom_artist_encoded = np.array([
    artist_encoder.transform([a])[0] if a in artist_encoder.classes_ else n_artists - 1
    for a in custom_df['artist_name']
])

custom_album_encoded = np.array([
    album_encoder.transform([al])[0] if al in album_encoder.classes_ else n_albums - 1
    for al in custom_df['album_name']
])

# Add encoded values to dataframe
custom_df['track_encoded'] = custom_track_encoded
custom_df['artist_encoded'] = custom_artist_encoded
custom_df['album_encoded'] = custom_album_encoded

# Drop encoded values since these are not in X train df during fit time
cols_to_use = [c for c in X_train.columns if c not in ['track_encoded', 'artist_encoded', 'album_encoded']]
X_custom_num = scaler.transform(
    custom_df.drop(['track_encoded', 'artist_encoded', 'album_encoded', 'track_name', 'track_id', 'artist_name', 'album_name', 'favor'], axis=1)
              .reindex(columns=cols_to_use, fill_value=0)
)

print("Transforming columns into tensors...")
X_custom_tensor = torch.tensor(X_custom_num, dtype=torch.float32)
custom_track_tensor = torch.tensor(custom_df['track_encoded'].values, dtype=torch.long)
custom_artist_tensor = torch.tensor(custom_df['artist_encoded'].values, dtype=torch.long)
custom_album_tensor = torch.tensor(custom_df['album_encoded'].values, dtype=torch.long)

# Run the model:
print("Running the model...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

print("Computing probabilities and predictions...")
with torch.no_grad():
    logits = model(X_custom_tensor, custom_track_tensor, custom_artist_tensor, custom_album_tensor)
    probabilities = torch.sigmoid(logits).view(-1)
    predictions = (probabilities > 0.5).int()

custom_df['favor_predictions'] = predictions.numpy()
custom_df['favor_probabilities'] = probabilities.numpy()

print("Saving results to CSV...")
custom_df.to_csv("model/results/custom_predictions.csv", index=False)

print("Done!")