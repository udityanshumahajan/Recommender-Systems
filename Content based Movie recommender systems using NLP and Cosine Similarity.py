#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import ast
ast.literal_eval


# In[31]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[32]:


movies.head(1)


# In[33]:


credits.head(1)


# In[34]:


movies = movies.merge(credits, on = 'title')


# In[35]:


#columns -  id, Keywords, title, overview, cast, crew


# In[36]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies


# In[37]:


movies.iloc[0].genres


# In[38]:


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[39]:


movies['genres'] = movies['genres'].apply(convert)


# In[40]:


movies['keywords']= movies['keywords'].apply(convert)


# In[41]:


movies['cast'][0]


# In[42]:


def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter = counter + 1
        else:
            break
    return l


# In[57]:


movies['cast'] = movies['cast'].apply(convert3)


# In[ ]:


def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


# In[ ]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[58]:


movies.head(1)


# In[59]:


def func1(x):
    if isinstance(x, str):
        return x.split()
    else:
        # Handle the case when 'x' is not a string (e.g., it's a float)
        return []

movies['overview'] = movies['overview'].apply(func1)


# In[60]:


movies


# In[65]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(' ', '') for i in x])


# In[67]:


movies.head()


# In[68]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[70]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[73]:


new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))


# In[75]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[76]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')


# In[78]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[86]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[88]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[90]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[91]:


from sklearn.metrics.pairwise import cosine_similarity


# In[96]:


similarity = cosine_similarity(vectors)


# In[104]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[105]:


recommend('Batman Begins')


# In[ ]:




