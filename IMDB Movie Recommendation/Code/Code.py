import pandas as pd
import numpy as np

colum_names = ['user_id', 'item_id', 'rating', 'timestamp']
user_data = pd.read_csv('Documents\\users.data', sep= '\t', names= colum_names)
movie_title = pd.read_csv('Documents\\movie_id_titles.csv')

df = pd.merge(user_data, movie_title, on=['item_id'])

rating_df = df.pivot_table(index='user_id', columns='title', values='rating')
user_rating = rating_df["Star Wars (1977)"]

similarity_movies = rating_df.corrwith(user_rating)

corr_movie_df = pd.DataFrame(similarity_movies, columns=['Correlation'])
corr_movie_df.dropna(inplace=True)

print(df.head())
