# Contextual Multi-Armed Bandits for Cold-Start Movie Recommendation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project tackles the **"Cold Start" problem** in recommender systems—how to recommend movies to new users with zero history. Unlike traditional Collaborative Filtering (which fails without data), we implemented **Contextual Multi-Armed Bandits (CMAB)** that learn user preferences in real-time by balancing exploration and exploitation.

We benchmarked three distinct architectures on the **MovieLens 100k** dataset:
1.  **LinUCB (Linear Upper Confidence Bound):** Deterministic optimism.
2.  **Linear Thompson Sampling (LinTS):** Probabilistic Bayesian exploration.
3.  **Stochastic Reward Environment (Novel Contribution):** A custom environment that assigns probabilistic rewards to "average" ratings (3-stars) to maximize user engagement.

## 🚀 Key Findings
Our experiments revealed a critical trade-off between **Convergence Speed** and **Recommendation Quality**:

| Algorithm | Role | Performance Highlight |
| :--- | :--- | :--- |
| **LinUCB** | **Speed Specialist** | Fastest convergence (**35.29% CTR**). Best for high-velocity streams. |
| **Thompson Sampling** | **Quality Specialist** | Superior List Quality (**27.10% Precision@5**). Best for "Top-N" lists. |
| **Stochastic Env(SE-LinUCB)** | **Engagement Engine** | Maximized total engagement volume (**53.42% CTR**) by capturing passive user interest. |

> **Note:** We also benchmarked against an offline **SVD (Collaborative Filtering)** model, which achieved 86.40% Precision@5, quantifying the "Cost of Cold Start" at approx. 59%.

## 🛠️ Algorithms Implemented
* **Contextual Bandits:** Disjoint LinUCB, Linear Thompson Sampling.
* **Environment:** Custom Gym-like environment for MovieLens data.
* **Reward Engineering:** Binary (Strict 4+ stars) vs. Stochastic (Probabilistic 3+ stars).
* **Baselines:** Random Agent, SVD (Matrix Factorization).

## 📊 Visualizations
*(Add your generated graphs here after uploading them to the repo)*

* **Cumulative CTR:** `images/ctr_graph.png`
* **Regret Analysis:** `images/regret_graph.png`
* **Precision@5 Comparison:** `images/precision_chart.png`

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install numpy pandas matplotlib scipy

```

### 2. Dataset
Download Link [GroupLens](https://grouplens.org "MovieLens Dataset Source")  


### 3. Running the Simulation
Execute the main script (or notebook) to run the 10,000-episode simulation:

```bash

python main.py
# Or run the Jupyter Notebook
jupyter notebook Bandit_Movie_Recommender.ipynb

```

📂 Project Structure

```
├── ml-100k/                # Dataset folder
├── images/                 # Saved graphs (CTR, Regret, Precision)
├── src/
│   ├── environment.py      # Custom MovieLens Environment
│   ├── agents.py           # LinUCB, Thompson Sampling Classes
│   └── evaluation.py       # Precision@K calculations
├── main.py                 # Main execution script
└── README.md               # Project documentation

```

---

🔮 Future Scope:

  * Hybrid Filtering: Transitioning from Bandits to Collaborative Filtering once sufficient user history is collected.

  * Neural Bandits: Implementing Deep Learning-based policies (NeuralUCB) to capture non-linear user-item relationships.

---

👥 Contributors:

  * **Rijaul Haque**

  * **Bhargab Kalita**

  * **Kangkita Baruah**

Dept. of Computer Science & Engineering,  
Dibrugarh University Institute of Engineering & Technology(DUIET)

---


📄 License:  

This project is licensed under the MIT License - see the LICENSE file for details.


---
