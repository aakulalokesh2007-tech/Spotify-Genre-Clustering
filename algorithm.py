import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Load the dataset

df = pd.read_csv('spotify dataset.csv')

#Clean the data (drop missing values and duplicates)
df = df.dropna()
df = df.drop_duplicates(subset=['track_id'])

# Define 'X' (the specific audio features we want the model to look at)
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
X = df[audio_features]


#Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Fit the K-Means Clustering Model
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

#Build the Recommendation Engine using Nearest Neighbors
nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
nn_model.fit(X_scaled)

# Generate a Recommendation

target_song_index = 0
distances, indices = nn_model.kneighbors([X_scaled[target_song_index]])

print(f"Recommendations for: {df.iloc[target_song_index]['track_name']} by {df.iloc[target_song_index]['track_artist']}")
print("-" * 50)

# Retrieve and print recommended songs (skipping index 0 since that's the target song itself)
for i in range(1, len(indices[0])):
    idx = indices[0][i]
    print(f"{i}. {df.iloc[idx]['track_name']} - {df.iloc[idx]['track_artist']}")
