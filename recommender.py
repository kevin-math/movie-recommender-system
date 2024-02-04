import os
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

def read_and_preprocess_data(file_path, nrows=None):
    df = pd.read_csv(file_path, nrows=nrows)
    df['movieId'] = df['movieId'].astype('int64')
    return df

def load_knn_model(file_path):
    return joblib.load(file_path)

def find_movie_id(movie_name, movie_metadata):
    best_match_tuple = process.extractOne(movie_name, movie_metadata['title'])
    best_match_index = best_match_tuple[2]
    return movie_metadata.loc[best_match_index, 'movieId']

def recommend_movies(cf_knn_model, user_item_matrix, movie_metadata, movie_id, n_recs=5):
    # Check if movie_id is a valid column in user_item_matrix
    if movie_id not in user_item_matrix.columns:
        raise ValueError(f"Column '{movie_id}' not found in user_item_matrix.")

    # Use kneighbors with the entire user-item matrix
    distances, indices = cf_knn_model.kneighbors(user_item_matrix, n_neighbors=10)
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    # List to store recommendations
    cf_recs = []
    for i in movie_rec_ids:
        cf_recs.append({'Title': movie_metadata['title'][i[0]], 'Distance': i[1]})

    # Select top number of recommendations needed
    df = pd.DataFrame(cf_recs[:n_recs], index=range(1, n_recs + 1))

    return df

# Read user ratings data
user_ratings_df = read_and_preprocess_data("./dataset/ratings.csv", nrows=20000)

# Read movie metadata and select relevant columns
movie_metadata = read_and_preprocess_data("./dataset/movies_metadata.csv", nrows=20000)
movie_metadata = movie_metadata[['title', 'genres', 'movieId']]

# Merge user ratings and movie metadata
movie_data = pd.merge(user_ratings_df, movie_metadata, on='movieId')

# Create user-item matrix
user_item_matrix = movie_data.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean').fillna(0)

# Define and save a KNN model on cosine similarity
cf_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

# Fit the KNN model on user-item matrix
cf_knn_model.fit(user_item_matrix)

# Example of extracting input movie ID and getting recommendations
movie_name = "Batman Returns"
input_movie_id = find_movie_id(movie_name, movie_metadata)
recommended_movies = recommend_movies(cf_knn_model, user_item_matrix, movie_metadata, input_movie_id)

# Display recommendations
print(recommended_movies)
