# Movie Recommendation System

## Overview

This Python script implements a simple movie recommendation system using collaborative filtering with k-nearest neighbors (KNN). The system leverages user ratings data and movie metadata to provide personalized movie recommendations based on user preferences and similarity between movies.

## Prerequisites

Before running the script, ensure that you have the required dependencies installed. You can install them using the following:

```bash
pip install pandas scikit-learn fuzzywuzzy
```

Dataset can be downloaded from -

https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download

## Usage

1. **Read and Preprocess Data**: The script reads user ratings data and movie metadata, preprocesses the data, and merges them based on the 'movieId' column.

2. **Create User-Item Matrix**: It generates a user-item matrix using the merged data, where rows represent users, columns represent movies, and values represent user ratings.

3. **Define and Train KNN Model**: The script uses scikit-learn's NearestNeighbors to define a KNN model with cosine similarity and fits it on the user-item matrix.

4. **Recommend Movies**: Given a movie name, the script identifies the corresponding movie ID using fuzzy string matching. Then, it recommends similar movies using the trained KNN model and displays the top recommendations.

## Script Structure

- `read_and_preprocess_data(file_path, nrows=None)`: Reads and preprocesses a CSV file containing movie data.
- `load_knn_model(file_path)`: Loads a pre-trained KNN model from a file.
- `find_movie_id(movie_name, movie_metadata)`: Finds the movie ID corresponding to a given movie name using fuzzy string matching.
- `recommend_movies(cf_knn_model, user_item_matrix, movie_metadata, movie_id, n_recs=5)`: Recommends movies based on the KNN model, user-item matrix, and a specified movie ID.

