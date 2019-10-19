import pandas as pd
import re

movies = pd.read_csv("movies.csv")
# Extract features
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])

# Calculate the distance
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recs(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices].tolist()
    
# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title, year = None):
    '''
    type title: str
    type year: str
    rtype: list[str]
    '''
   
    def titleSearch(title):
        '''
        rtype: pandas.series
        '''
        t = titles.str.contains(title, case = False)
        match = titles[t]
        return match.tolist()
    
    def recommend(title, year):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # recommend 5 movies
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        movies = titles.iloc[movie_indices].tolist()
        
        # match the year
        if year:
            movies = [s for s in movies if year in s]
                
        return movies
    
    match = titleSearch(title)
    rec = {}
    for t in match:
        rec[t] = recommend(t,year)
        
    return match, rec

#genre_recommendations('Saving Private Ryan')
#genre_recommendations('Toy Story')
