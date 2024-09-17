# views.py
from django.shortcuts import render
from .models import Movie
from difflib import get_close_matches # For handling potential typos in movie titles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import difflib

def startPage(request):
    return render(request,'index.html')

def index2(request):
    return render(request,'index2.html')

def get_movie_recommendations(movie_title):
    new_df = pd.read_csv('new_movies_data_with_tags1.csv')
    # Check if 'tags' column exists and contains strings before applying lower()
    if 'tags' in new_df.columns and new_df['tags'].dtype == 'object': 
       new_df['tags'] = new_df['tags'].apply(lambda x:x.lower() if isinstance(x, str) else x)
    
    new_df.to_csv('main_data.csv')
    import nltk
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    def stem(text):
        # Handle non-string values in 'text'
        if not isinstance(text, str): 
            return ""  # Or any suitable default value for non-string entries
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    new_df['tags'] = new_df['tags'].apply(stem)
    new_df.head(2)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    vectors[0]
    vectors.shape
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(vectors)
    similarity
    movie_index = new_df[new_df['title'] == movie_title].index
    if len(movie_index) == 0:
        return "No movie found"
    movie_index = movie_index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    result = []
    for i in movies_list:
        poster_url = new_df.iloc[i[0]].poster_url
        movie_title = new_df.iloc[i[0]].title
        ott = new_df.iloc[i[0]].OTT
        result.append((movie_title, poster_url,ott))
    print(result[0])
    return result

def get_genre_recommendations(genre, num_recommendations=5):
    movies_data = pd.read_csv('new_movies_data1.csv')
    movies_data['genres'] = movies_data['genres'].fillna('Unknown')
    combined_features = movies_data['genres']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    list_of_all_genres = movies_data['genres'].tolist()
    
    find_close_match = difflib.get_close_matches(genre,list_of_all_genres)
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.genres == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))

    sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)
    
    result = []
    for i, j in enumerate(sorted_similar_movies[:5]): 
        movie_id = movies_data.iloc[j[0]].movie_id
        title_from_index = movies_data.iloc[j[0]].title
        poster_url = movies_data.iloc[j[0]].poster_url
        ott = movies_data.iloc[j[0]].OTT

        # Fetch the image and encode it as base64
        

        result.append((title_from_index,poster_url,ott))

    return result

def movie_recommendations(request):
    if request.method == 'POST':
        search_type = request.POST.get('search_type') 
        if search_type == 'title':
            search_title = request.POST.get('movie_title')
            print(f"Search title: {search_title}")
            # Handle potential typos in the movie title (same as before)
            all_titles = Movie.objects.values_list('title', flat=True)
            close_matches = get_close_matches(search_title, all_titles, n=1, cutoff=0.6)
            if close_matches:
                search_title = close_matches[0]
            recommendations = get_movie_recommendations(search_title)
            print(recommendations)
            print(type(recommendations))
        elif search_type == 'genre':
            search_genre = request.POST.get('genre')
            print(f"Search genre: {search_genre}")
            recommendations = get_genre_recommendations(search_genre)
            print(recommendations)
        else:
            recommendations = []  # Handle invalid search type
        
        return render(request, 'movie_recommendations.html', {'recommendations': recommendations})
    else:
        return render(request, 'movie_search.html')