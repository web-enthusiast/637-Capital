import pandas as pd

file_path = '/Users/ADMIN/Desktop/movie/data.txt'

data = pd.read_csv('/Users/ADMIN/Desktop/movie/data.txt')
print('Original Data:')
print(data.head())

#clean data
def clean_data(data):
    data = data.drop_duplicates(inplace=True)
    cleaned_file_path = '/Users/ADMIN/Desktop/movie/data.txt'
  # Remove rows with missing or invalid data
    data = data.dropna(subset=['User', 'Movie', 'Rating'], how='any')
    data = data[~data['Rating'].str.contains('[a-zA-Z]')]  
    data['Rating'] = data['Rating'].astype(float)
    return data

cleaned_data = clean_data(data)
print("cleaned data:")
print(data.head())

  

# Data Analysis
movie_ratings = cleaned_data.groupby('Movie')['Rating'].agg(['mean', 'count']).reset_index()

#Recommendation algorithm
def recommend_movies(user, data, fallback=True):
    # Filter data for the given user
    user_data = data[data['User'] == user]
    
    if len(user_data) == 0 or user_data['Rating'].isnull().all():
        if fallback:
            # Fallback: Recommend popular movies if no user data or incomplete data
            popular_movies = movie_ratings.sort_values(by=['count', 'mean'], ascending=False)
            return popular_movies.head(5)['Movie'].tolist()
        else:
            return "Not enough data to make recommendations."

 # Create a user-item matrix
    user_movie_matrix = pd.pivot_table(data, values='Rating', index='User', columns='Movie', fill_value=0)
    
    tfidf = TfidfVectorizer(stop_words='english')
    movie_matrix = tfidf.fit_transform(user_movie_matrix.columns)
    cosine_sim = linear_kernel(movie_matrix, movie_matrix)
    
    # Get movies similar to the ones the user has liked
    similar_movies = []
    for movie in user_data['Movie']:
        movie_index = user_movie_matrix.columns.get_loc(movie)
        sim_scores = list(enumerate(cosine_sim[movie_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  
        movie_indices = [i[0] for i in sim_scores]
        similar_movies.extend(user_movie_matrix.columns[movie_indices])
    
    # Recommend unique movies 
    recommended_movies = list(set(similar_movies) - set(user_data['Movie']))
    return recommended_movies[:5]

# Command Line Interface (CLI)
def display_recommendations(recommendations):
    if isinstance(recommendations, list):
        print("Recommended movies:")
        for idx, movie in enumerate(recommendations, start=1):
            print(f"{idx}. {movie}")
    else:
        print(recommendations)

def user_interface():
    username = input("Enter your username: ")
    recommendations = recommend_movies(username, cleaned_data)
    display_recommendations(recommendations)

user_interface()