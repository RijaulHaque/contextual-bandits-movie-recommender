# File: contextual-bandits-movie-recommender/main.py

import numpy as np
import matplotlib.pyplot as plt

# Import from our custom modules
from src.environment import load_movielens_data, MovieLensBanditEnv, StochasticMovieLensEnv
from src.agents import LinUCBAgent, LinearTSAgent
from src.evaluation import (
    calculate_precision_at_k, 
    calculate_ts_precision_at_k, 
    run_svd_benchmark,
    plot_cumulative_ctr,
    plot_cumulative_regret,
    plot_precision_comparison
)

# ------------------------------------------------------------------------------------
# HELPER FUNCTION TO RUN SIMULATIONS
# ------------------------------------------------------------------------------------
def run_simulation(agent, env, num_episodes, label="Agent"):
    """
    Runs a simulation loop for a specific agent and environment.
    """
    print(f"\n--- Starting {label} Simulation ({num_episodes} episodes) ---")
    clicks = 0
    rewards_history = []
    
    for i in range(num_episodes):
        # 1. Get Context
        user_context, _ = env.reset()
        
        # 2. Agent Selects Arm
        action = agent.select_arm(user_context)
        
        # 3. Environment Step
        _, reward, _, _, _ = env.step(action)
        
        # 4. Agent Updates Knowledge
        agent.update(action, user_context, reward)
        
        # 5. Track Metrics
        clicks += int(reward)
        rewards_history.append(reward)
        
        if (i + 1) % 1000 == 0:
            current_ctr = (clicks / (i + 1)) * 100
            print(f"Episode {i+1}: CTR = {current_ctr:.2f}%")
            
    final_ctr = (clicks / num_episodes) * 100
    print(f"Completed {label}. Final CTR: {final_ctr:.2f}%")
    return rewards_history, final_ctr

# ------------------------------------------------------------------------------------
# MAIN EXECUTION FLOW
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- Step 1: Data Loading ---
    print("\n[1/7] Loading MovieLens Data...")
    user_features, rating_lookup = load_movielens_data()
    
    # Constants
    NUM_EPISODES = 10000
    N_TEST_USERS = 940 # Validation set size
    
    # --- Step 2: Initialize Environments ---
    print("\n[2/7] Initializing Environments...")
    env_standard = MovieLensBanditEnv(user_features, rating_lookup)
    env_stochastic = StochasticMovieLensEnv(user_features, rating_lookup)
    
    n_arms = env_standard.action_space.n
    n_features = env_standard.observation_space.shape[0]

    # --- Step 3: Run Random Agent Baseline ---
    print("\n[3/7] Running Random Agent Baseline...")
    random_clicks = 0
    for _ in range(NUM_EPISODES):
        env_standard.reset()
        _, r, _, _, _ = env_standard.step(env_standard.action_space.sample())
        random_clicks += r
    random_ctr = (random_clicks / NUM_EPISODES) * 100
    print(f"Random Agent CTR: {random_ctr:.2f}%")

    # --- Step 4: Run Bandit Simulations ---
    print("\n[4/7] Running Bandit Algorithms...")
    
    # A. LinUCB (Standard Env)
    agent_linucb = LinUCBAgent(n_arms, n_features, alpha=0.1)
    linucb_rewards, _ = run_simulation(agent_linucb, env_standard, NUM_EPISODES, "LinUCB")

    # B. Thompson Sampling (Standard Env)
    agent_ts = LinearTSAgent(n_arms, n_features, v=0.1)
    ts_rewards, _ = run_simulation(agent_ts, env_standard, NUM_EPISODES, "Thompson Sampling")

    # C. Stochastic LinUCB (Stochastic Env)
    agent_stoch = LinUCBAgent(n_arms, n_features, alpha=0.1)
    stoch_rewards, _ = run_simulation(agent_stoch, env_stochastic, NUM_EPISODES, "Stochastic Agent")

    # --- Step 5: Plot Learning Curves ---
    print("\n[5/7] Generating Learning Curves...")
    plot_cumulative_ctr(linucb_rewards, ts_rewards, stoch_rewards, random_ctr)
    plot_cumulative_regret(linucb_rewards, ts_rewards, stoch_rewards)

    # --- Step 6: Evaluate Recommendation Quality (Precision@5) ---
    print("\n[6/7] Evaluating Recommendation Quality (Precision@5)...")
    
    # A. LinUCB P@5
    p5_linucb = calculate_precision_at_k(agent_linucb, env_standard, k=5, n_test_users=N_TEST_USERS)
    print(f"LinUCB Precision@5: {p5_linucb*100:.2f}%")
    
    # B. Thompson Sampling P@5
    p5_ts = calculate_ts_precision_at_k(agent_ts, env_standard, k=5, n_test_users=N_TEST_USERS)
    print(f"Thompson Sampling Precision@5: {p5_ts*100:.2f}%")
    
    # C. Stochastic Agent P@5
    # Note: We test the stochastic agent on the STANDARD environment to check strict quality
    p5_stoch = calculate_precision_at_k(agent_stoch, env_standard, k=5, n_test_users=N_TEST_USERS)
    print(f"Stochastic Agent Precision@5: {p5_stoch*100:.2f}%")
    
    # D. SVD Benchmark (Collaborative Filtering)
    p5_svd = run_svd_benchmark(k=5, n_test_users=N_TEST_USERS)
    print(f"SVD (Benchmark) Precision@5: {p5_svd*100:.2f}%")

    # --- Step 7: Final Comparison Plot ---
    print("\n[7/7] Generating Final Comparison Plot...")
    
    metrics = {
        'Random': random_ctr, # Random P@5 is roughly equal to its CTR
        'LinUCB': p5_linucb * 100,
        'Thompson Sampling': p5_ts * 100,
        'Stochastic': p5_stoch * 100,
        'SVD (CF)': p5_svd * 100
    }
    
    plot_precision_comparison(metrics)
    
    print("\n--- Experiment Complete! ---")