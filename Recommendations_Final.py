from json import load
from numpy import sort
import pandas as pd
import pickle
import random
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
import numpy as np

class NewUser():
    
    def __init__(self, ratings, movies, k):
        """ k: number of movies for random rating of a new user"""

        self.ratings = ratings
        self.movies = movies
        self.k = k
        self.get_movies = []
        self.Ratings = []
        self.get_ratings = {}
    
    def rating_transformation(self):
        
        movies_w_rating = pd.merge(self.movies,self.ratings, left_on='movieId', right_on='movieId',)
        #print(movies_w_rating.head())
        movies_w_rating = movies_w_rating.drop(['genres','timestamp','movieId'], axis=1)
        #print(movies_w_rating.head(10))
        ratings_tab = pd.pivot_table(movies_w_rating,
                                    index='userId',
                                    columns='title',
                                    values = 'rating')
        #print(ratings_tab)
        imputer = KNNImputer(n_neighbors=5)
        ratings_tab = pd.DataFrame(imputer.fit_transform(ratings_tab),
                            index = ratings_tab.index,
                            columns = ratings_tab.columns)
        #print(ratings_tab)
        return ratings_tab

    def get_movies_rand(self,ratings_tab):

        new_user = np.zeros((1, len(ratings_tab.columns)), dtype=int).tolist()[0]
        for i in range(self.k):
            random_movies = random.choice(self.movies['title'])
            self.get_movies.append(random_movies)
            self.Ratings.append(random.randint(1,5))
        for movie, rating in zip(self.get_movies, self.Ratings):
            mov_rating = {movie : rating}
            self.get_ratings.update(mov_rating)
        #print(get_ratings)
        for index, item in enumerate(ratings_tab.columns):
            if item in self.get_ratings.keys():
                new_user[index] = self.get_ratings[item]
        new_user  = pd.DataFrame([new_user], index = ['actual_rating'], columns = ratings_tab.columns)
        return new_user
        
    def transpose_new_user(self,new_user):
        
        new_user_T = new_user.T
        new_user_T.reset_index(inplace=True)
        return new_user_T

    
class RecommendMovie(NewUser):
    """ m is number of movies a user want to be recommended"""

    def __init__(self, k, m, loaded_model):
        super().__init__(ratings, movies, k)
        self.loaded_model = loaded_model
        self.m = m

    def new_recommendations(self):

        new_recommendation = self.loaded_model.transform(super().get_movies_rand(super().rating_transformation()))
        new_comp = self.loaded_model.components_.T
        new_rating_pred = new_comp.dot(new_recommendation.T).T
        new_rating_pred = pd.DataFrame(new_rating_pred, columns = super().get_movies_rand(super().rating_transformation()).columns).T #columns=ratings.columns
        new_rating_pred.reset_index(inplace=True)
        new_rating_pred.columns = ['title','predicted_rating']
        new_rating_pred['actual_rating'] = super().transpose_new_user(super().get_movies_rand(super().rating_transformation()))['actual_rating']
        return new_rating_pred

    def recommend_movie_activeuser(self, new_rating_pred ):
    
        rec_user_id = new_rating_pred 
        rec_movie = rec_user_id[rec_user_id['actual_rating'] == 0.0]
        Recommended_Movies =rec_movie.sort_values(by = 'predicted_rating', ascending = False)[:self.m]
        Movies = Recommended_Movies['title']
        print(Movies)

# load the model
loaded_model = pickle.load(open('model_ten.bin', 'rb'))

# read the .csv file
movies = pd.read_csv('./data/ml-latest-small/movies.csv')
ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')

# create an object
jayesh_recommends = RecommendMovie(100, 10, loaded_model)
jayesh_recommends.recommend_movie_activeuser(jayesh_recommends.new_recommendations())
    
















# new_recommendation = loaded_model.transform(new_user_)
# #print(new_recommendation)
# new_comp = loaded_model.components_.T
# #print(new_comp)
# new_rating_pred = new_comp.dot(new_recommendation.T).T
# new_rating_pred = pd.DataFrame(new_rating_pred, columns = new_user_.columns).T #columns=ratings.columns
# new_rating_pred.reset_index(inplace=True)
# new_rating_pred.columns = ['title','predicted_rating']
# #print(new_rating_pred)

# new_rating_pred['actual_rating'] = new_user_T_['actual_rating']
# #print(new_rating_pred)
# def recommend_movie_activeuser(prediction_df,m):

#     rec_user_id = prediction_df 
#     rec_movie = rec_user_id[rec_user_id['actual_rating'] == 0.0]
#     Recommended_Movies =rec_movie.sort_values(by = 'predicted_rating', ascending = False)[:m]
#     Movies = Recommended_Movies['title']
#     print(Movies)

# recommend_movie_activeuser(new_rating_pred, m = 5)







































