# File: contextual-bandits-movie-recommender/src/environment.py

###########################         DEPENDENCIES         ##############################

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests, zipfile, io
import os
import warnings
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Suppress the harmless warnings to keep output clean
warnings.filterwarnings("ignore")

###################         DATA LOADING UTILS        ###################

def load_movielens_data():
    """
    Downloads and preprocesses the MovieLens 100k dataset.
    Returns:
        user_features (dict): Preprocessed user vectors.
        rating_lookup (dict): Dictionary of user-item ratings.
    """
    # 1. Download Data (Only if not present)
    if not os.path.exists('ml-100k'):
        print("--- Step 1: Downloading MovieLens Data ---")
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        try:
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(".") # Extracts to a folder named 'ml-100k'
        except Exception as e:
            print(f"Error downloading data: {e}")
            return {}, {}
    else:
        print("--- Data already exists. Loading local files. ---")

    # 2. Load Ratings (User-Item Interactions)
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

    # 3. Load Users (Demographics)
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

    print(f"Loaded {len(ratings)} ratings and {len(users)} users.")

    print("--- Step 2: Preprocessing Data ---")
    
    # Create Rating Lookup
    rating_lookup = {}
    for _, row in ratings.iterrows():
        u_id = int(row['user_id'])
        m_id = int(row['movie_id'])
        rating_lookup.setdefault(u_id, {})[m_id] = row['rating']

    # Create User Context Vectors
    users['age_norm'] = users['age'] / users['age'].max()
    users['gender_binary'] = users['gender'].apply(lambda x: 1 if x == 'M' else 0)
    occupation_dummies = pd.get_dummies(users['occupation'], prefix='occ', dtype=float)
    users_encoded = pd.concat([users, occupation_dummies], axis=1)

    user_features = {}
    feature_cols = ['age_norm', 'gender_binary'] + list(occupation_dummies.columns)

    for _, row in users_encoded.iterrows():
        u_id = int(row['user_id'])
        user_features[u_id] = row[feature_cols].values.astype(np.float32)

    print(f"User Features Created. Dimension size: {len(feature_cols)}")
    
    return user_features, rating_lookup

###################         GYMNASIUM ENVIRONMENTS        ###################

class MovieLensBanditEnv(gym.Env):
    """
    Standard Deterministic Environment.
    Reward = 1 if Rating >= 4, else 0.
    """
    def __init__(self, user_features, rating_lookup):
        super().__init__()

        self.user_features = user_features
        self.rating_lookup = rating_lookup

        # Action Space: Recommend a Movie ID (1 to 1682)
        self.num_movies = 1683
        self.action_space = spaces.Discrete(self.num_movies)

        # Observation Space: The User Feature Vector
        if len(user_features) > 0:
            obs_dim = len(list(user_features.values())[0])
            self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.current_user_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random user
        user_ids = list(self.user_features.keys())
        self.current_user_id = self.np_random.choice(user_ids)

        return self.user_features[self.current_user_id], {}

    def step(self, action):
        # Check Reward: Did the user rate this movie?
        user_history = self.rating_lookup.get(self.current_user_id, {})
        true_rating = user_history.get(action, 0)

        # Reward Logic: 1 if rating >= 4 (Like), else 0
        reward = 1.0 if true_rating >= 4 else 0.0

        return self.user_features[self.current_user_id], reward, True, False, {}


class StochasticMovieLensEnv(gym.Env):
    """
    Stochastic Environment for Reward Shaping.
    Assigns probabilistic rewards to 3-star ratings to mimic user passivity.
    """
    def __init__(self, user_features, rating_lookup):
        super().__init__()
        self.user_features = user_features
        self.rating_lookup = rating_lookup

        # Standard Action Space (1683 Movies)
        self.num_movies = 1683
        self.action_space = spaces.Discrete(self.num_movies)

        # Observation Space
        if len(user_features) > 0:
            obs_dim = len(list(user_features.values())[0])
            self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        else:
             self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.current_user_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        user_ids = list(self.user_features.keys())
        self.current_user_id = self.np_random.choice(user_ids)
        return self.user_features[self.current_user_id], {}

    def step(self, action):
        # 1. Look up the true rating
        user_history = self.rating_lookup.get(self.current_user_id, {})
        true_rating = user_history.get(action, 0)

        # 2. Define Probabilities (The Stochastic Logic)
        prob_click = 0.0

        if true_rating == 5:
            prob_click = 1.0    # Guaranteed Click
        elif true_rating == 4:
            prob_click = 0.95   # 95% chance
        elif true_rating == 3:
            prob_click = 0.45   # 45% chance (The Exploration Zone)
        else:
            prob_click = 0.0    # 1 or 2 stars = No chance

        # 3. Roll the Dice
        if np.random.rand() < prob_click:
            reward = 1.0
        else:
            reward = 0.0

        return self.user_features[self.current_user_id], reward, True, False, {}
    

# ------------------------------------------------------------------------------------
