# File: contextual-bandits-movie-recommender/src/evaluation.py

###########################         DEPENDENCIES         ##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

#------------------------------------------------------------------------------------------

###################         METRIC CALCULATION FUNCTIONS        ###################

def calculate_precision_at_k(agent, env, k=5, n_test_users=1000):
    """
    Calculates Precision@K for a standard Deterministic Agent (LinUCB).
    """
    cumulative_precision = 0
    print(f"Testing Precision@{k} on {n_test_users} users...")
    
    for _ in range(n_test_users):
        user_context, _ = env.reset()
        user_id = env.current_user_id
        
        # Get scores for all arms
        scores = []
        x = user_context.reshape(-1, 1)
        
        for a in range(agent.n_arms):
            A_inv = np.linalg.inv(agent.A[a])
            theta = np.dot(A_inv, agent.b[a])
            
            # UCB Score
            expected_reward = np.dot(theta.T, x).item()
            uncertainty = agent.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)).item())
            scores.append(expected_reward + uncertainty)
            
        # Select Top K
        top_k_movies = np.argsort(scores)[-k:][::-1]
        
        # Check Ground Truth
        hits = 0
        user_history = env.rating_lookup.get(user_id, {})
        for movie_id in top_k_movies:
            if user_history.get(movie_id, 0) >= 4:
                hits += 1
        
        cumulative_precision += (hits / k)

    return cumulative_precision / n_test_users

def calculate_ts_precision_at_k(agent, env, k=5, n_test_users=1000):
    """
    Calculates Precision@K for a Thompson Sampling Agent (using Greedy Mean).
    """
    cumulative_precision = 0
    print(f"Testing TS Precision@{k} on {n_test_users} users...")
    
    for _ in range(n_test_users):
        user_context, _ = env.reset()
        user_id = env.current_user_id
        x = user_context.reshape(-1, 1)
        scores = []
        
        for a in range(agent.n_arms):
            # Use Mean (Greedy) for evaluation stability
            A_inv = np.linalg.inv(agent.A[a])
            theta_hat = np.dot(A_inv, agent.b[a])
            mean_score = np.dot(theta_hat.T, x).item()
            scores.append(mean_score)
            
        top_k_movies = np.argsort(scores)[-k:][::-1]
        
        hits = 0
        user_history = env.rating_lookup.get(user_id, {})
        for movie_id in top_k_movies:
            if user_history.get(movie_id, 0) >= 4:
                hits += 1
        
        cumulative_precision += (hits / k)

    return cumulative_precision / n_test_users

def run_svd_benchmark(k=5, n_test_users=1000, data_path='ml-100k/u.data'):
    """
    Trains an offline SVD model and calculates its Precision@K (Theoretical Ceiling).
    """
    print("\n--- Running Collaborative Filtering (SVD) Benchmark ---")
    
    # 1. Load Data
    cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    df = pd.read_csv(data_path, sep='\t', names=cols)
    
    # 2. Matrix Factorization
    R_df = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    # SVD
    U, sigma, Vt = svds(R_demeaned, k=50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns, index=R_df.index)
    
    # 3. Evaluation
    cumulative_precision = 0
    ground_truth = df[df['rating'] >= 4].groupby('user_id')['movie_id'].apply(set).to_dict()
    test_users = list(preds_df.index)[:n_test_users]
    
    for user_id in test_users:
        user_preds = preds_df.loc[user_id]
        top_k_movies = user_preds.nlargest(k).index.tolist()
        
        hits = 0
        actual_likes = ground_truth.get(user_id, set())
        for movie_id in top_k_movies:
            if movie_id in actual_likes:
                hits += 1
        cumulative_precision += (hits / k)
        
    return cumulative_precision / len(test_users)

###################         PLOTTING FUNCTIONS        ###################

def plot_cumulative_ctr(linucb_rewards, ts_rewards, stoch_rewards, random_ctr):
    """
    Plots the learning curves (CTR) for all agents.
    """
    # Calculate Cumulative CTRs
    ctr_linucb = np.cumsum(linucb_rewards) / (np.arange(len(linucb_rewards)) + 1) * 100
    ctr_ts = np.cumsum(ts_rewards) / (np.arange(len(ts_rewards)) + 1) * 100
    ctr_stoch = np.cumsum(stoch_rewards) / (np.arange(len(stoch_rewards)) + 1) * 100
    
    final_linucb = ctr_linucb[-1]
    final_ts = ctr_ts[-1]
    final_stoch = ctr_stoch[-1]

    plt.figure(figsize=(12, 6))
    plt.plot(ctr_linucb, label=f'LinUCB (Final: {final_linucb:.2f}%)', color='blue')
    plt.plot(ctr_ts, label=f'Thompson Sampling (Final: {final_ts:.2f}%)', color='green')
    plt.plot(ctr_stoch, label=f'Stochastic Env (Final: {final_stoch:.2f}%)', color='orange')
    plt.axhline(y=random_ctr, color='red', linestyle='--', label=f'Random Baseline {random_ctr:.2f}%')

    plt.title('Algorithms Comparison: Cumulative CTR')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative CTR (%)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_cumulative_regret(linucb_rewards, ts_rewards, stoch_rewards):
    """
    Plots the cumulative regret (Lost Clicks) for all agents.
    Regret = 1 - Reward (Assuming optimal reward is always 1).
    """
    regret_linucb = np.cumsum([1 - r for r in linucb_rewards])
    regret_ts = np.cumsum([1 - r for r in ts_rewards])
    regret_stoch = np.cumsum([1 - r for r in stoch_rewards])

    plt.figure(figsize=(12, 6))
    plt.plot(regret_ts, label='Thompson Sampling', color='green')
    plt.plot(regret_linucb, label='LinUCB', color='blue')
    plt.plot(regret_stoch, label='Stochastic Env', color='orange')
    plt.axhline(y=0, color='black', linestyle='--', label='Perfect Agent')

    plt.title('Cumulative Regret Analysis (Lower is Better)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Lost Clicks (Regret)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_precision_comparison(metrics_dict):
    """
    Plots a bar chart comparing Precision@K values.
    metrics_dict = {'Algorithm Name': value_in_percent, ...}
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color=colors[:len(names)], alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Recommendation Quality: Cold Start (Bandits) vs Warm Start (CF)', fontsize=14)
    plt.ylabel('Precision@5 (%)', fontsize=12)
    plt.ylim(0, 100) 
    plt.grid(axis='y', alpha=0.3)
    plt.show()

#------------------------------------------------------------------------------------------