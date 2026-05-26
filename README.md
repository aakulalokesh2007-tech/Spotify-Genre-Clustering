# 🎵 Spotify Genre Grouping & Recommendation System

## 📌 Project Overview
This project applies Machine Learning to automatically segment Spotify songs into clusters based on their auditory features (like energy, acousticness, and danceability) without relying on human-made genre tags. Using these groupings, I built a content-based recommendation engine that suggests similar songs to users.

## 🛠️ Tools & Technologies Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Algorithms:** K-Means Clustering, Principal Component Analysis (PCA), Nearest Neighbors (Cosine Similarity)

## 📊 Key Insights & Visualizations
1. **Feature Correlation:** Found strong positive correlations between *Energy* & *Loudness*, and negative correlations between *Energy* & *Acousticness*.
2. **K-Means Clustering:** Applied the Elbow Method to determine optimal clusters ($k=6$). Visualized the clusters using PCA to reduce the 9-dimensional audio data into a 2D plot.
3. **Recommendation Engine:** Utilized Nearest Neighbors to recommend top 5 similar tracks based on the multi-dimensional audio profile of a target song.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed (`pip install pandas numpy scikit-learn matplotlib seaborn`).
3. Run `algorithm.py`.

## 📈 Sample Output
*Target Song: "I Don't Care (with Justin Bieber) - Loud Luxury Remix" by Ed Sheeran*
**Recommendations:**
1. I Miss You (feat. Julia Michaels) - Cahill Remix by Clean Bandit
2. Rise - TV Noise Ibiza Mix by Jonas Blue
3. Name & Number by Shift K3Y
