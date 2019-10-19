
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

### Import data

movies = pd.read_csv("movies.csv")
movies.head()

### Genres
# Extract features
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])

# Calculate the distance
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])


### Tag
tags = pd.read_csv("tags.csv")
del tags['timestamp']
tags = tags.dropna()
tags['tag'] = tags['tag'].str.lower()
tags['tag'] = tags['tag'].apply(lambda t: t.translate(str.maketrans('', '', string.punctuation)))
tags = tags.groupby('movieId')['tag'].apply(lambda t: ' '.join(t))
#tags.head()


# Function that get movie recommendations based on the cosine similarity score of movie genres
def recommendations(title, tag = None, year = None):
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
    
    
    def recommend(title, year, tag):
        '''
        rtype: list
        '''
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # recommend movies
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        
        # get the movie id
        movie_id = movies['movieId'].iloc[movie_indices].tolist()
        movie_rec = titles.iloc[movie_indices].tolist()
        
        # filter with tag
        if tag:
            tag = tag.lower()
            movie_tag = tags[movie_id]
            sel = [k for k,t in enumerate(movie_tag.dropna()) if tag in t]
            movie_id = [movie_id[i] for i in sel]
            movie_rec = [movies['title'][movies['movieId'] == i].tolist()[0] for i in movie_id]
        
        # match the year
        if year:
            movie_rec = [s for s in movie_rec if year in s]
                
        return movie_rec
    
    name = titleSearch(title)
    
    lst = []
    for t in name:
        lst += recommend(t,year,tag)
        
    return name, lst


# store the possible titles and the recommended movies
match, recommend_movies = recommendations('Toy Story', 'pixar','2011')


# match
# recommend_movies
