# File: contextual-bandits-movie-recommender/src/agents.py

###########################         DEPENDENCIES         ##############################

import numpy as np

#------------------------------------------------------------------------------------------

###################         CONTEXTUAL BANDIT AGENTS        ###################

class LinUCBAgent:
    """
    Linear Upper Confidence Bound (LinUCB) Agent (Disjoint Model).
    
    Attributes:
        alpha (float): Exploration parameter.
        A (list): List of covariance matrices (one per arm).
        b (list): List of reward vectors (one per arm).
    """
    def __init__(self, n_arms, n_features, alpha=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Initialize A (Identity) and b (Zeros) for each arm
        # A represents the covariance (uncertainty)
        # b represents the weighted history of rewards
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context_vector):
        """
        Selects the best arm based on UCB score.
        Formula: prediction + alpha * uncertainty
        """
        p_values = []
        x = context_vector.reshape(-1, 1)

        for a in range(self.n_arms):
            # 1. Inverse Covariance (Ridge Regression)
            # In production, use Sherman-Morrison for O(d^2) updates instead of O(d^3) inversion
            A_inv = np.linalg.inv(self.A[a])

            # 2. Weight Estimate (Theta)
            theta = np.dot(A_inv, self.b[a])

            # 3. UCB Calculation
            expected_reward = np.dot(theta.T, x).item()
            uncertainty = self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)).item())
            
            p_values.append(expected_reward + uncertainty)

        return np.argmax(p_values)

    def update(self, arm, context_vector, reward):
        """
        Updates the A matrix and b vector for the chosen arm.
        """
        x = context_vector.reshape(-1, 1)
        self.A[arm] += np.dot(x, x.T)
        self.b[arm] = self.b[arm].reshape(-1, 1) + reward * x



#------------------------------------------------------------------------------------------

class LinearTSAgent:
    """
    Linear Thompson Sampling (LinTS) Agent.
    
    Attributes:
        v (float): Variance parameter controlling exploration.
    """
    def __init__(self, n_arms, n_features, v=0.01):
        self.n_arms = n_arms
        self.n_features = n_features
        self.v = v 

        # Initialize A and b (Shared logic with LinUCB)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context_vector):
        """
        Selects an arm by sampling from the posterior distribution.
        """
        x = context_vector.reshape(-1, 1)
        sampled_scores = []

        for a in range(self.n_arms):
            # 1. Posterior Calculation
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = np.dot(A_inv, self.b[a])

            # 2. Sampling parameters
            mean_score = np.dot(theta_hat.T, x).item()
            # Variance scaled by v parameter
            variance_score = self.v**2 * np.dot(x.T, np.dot(A_inv, x)).item()
            std_dev = np.sqrt(variance_score)

            # 3. Sample from Normal Distribution
            sample = np.random.normal(mean_score, std_dev)
            sampled_scores.append(sample)

        return np.argmax(sampled_scores)

    def update(self, arm, context_vector, reward):
        """
        Updates the parameters (Identical to LinUCB update).
        """
        x = context_vector.reshape(-1, 1)
        self.A[arm] += np.dot(x, x.T)
        self.b[arm] = self.b[arm].reshape(-1, 1) + reward * x



#------------------------------------------------------------------------------------------