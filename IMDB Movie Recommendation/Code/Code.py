import pandas as pd
import numpy as np

colum_names = ['user_id', 'item_id', 'rating', 'timestammp']
user_data = pd.read_csv('Documents\\users.data', sep= '\t', names= colum_names)
movie_title = pd.read_csv('Documents\\movie_id_titles.csv')

df = pd.merge(user_data, movie_title, on=['item_id'])

rating_df = df.pivot_table(index='user_id', columns='title', values='rating')
print(rating_df.head())
user_rating = rating_df["Mask, The (1994)"]
print(user_rating.head())