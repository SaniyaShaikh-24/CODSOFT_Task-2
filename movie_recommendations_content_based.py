from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample movie data: movie and its genres
movies = {
    'Inception': 'Action, Sci-Fi, Thriller',
    'Interstellar': 'Adventure, Drama, Sci-Fi',
    'The Dark Knight': 'Action, Crime, Drama',
    'Forrest Gump': 'Drama, Romance',
    'Titanic': 'Drama, Romance'
}

# Function to recommend movies based on movie genres
def recommend_movies(movie_data, target_movie):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_data.values())
    
    # Compute cosine similarity between movies
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the target movie
    target_index = list(movie_data.keys()).index(target_movie)
    
    # Get movie similarities with the target movie
    movie_similarities = list(enumerate(cosine_sim[target_index]))
    
    # Sort movies by similarity in descending order
    movie_similarities = sorted(movie_similarities, key=lambda x: x[1], reverse=True)
    
    # Exclude the target movie and get top recommendations
    recommendations = [(list(movie_data.keys())[i], sim) for i, sim in movie_similarities if i != target_index]
    
    return recommendations

# Example: Recommend movies similar to 'Inception'
target_movie = 'Inception'
similar_movies = recommend_movies(movies, target_movie)
print(f"Movies similar to '{target_movie}': {similar_movies}")
