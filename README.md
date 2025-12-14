# Comparative Analysis of Contextual Multi-Armed Bandits for Cold-Start Movie Recommendation

## 📌 Project Overview
This repository contains the source code and datasets for the research paper **"Comparative Analysis of Contextual Multi-Armed Bandits for Cold-Start Movie Recommendation"**.

The project addresses the "Cold Start" problem in recommender systems—recommending items to new users with no prior history—by implementing and benchmarking three Contextual Multi-Armed Bandit (CMAB) algorithms on the MovieLens 100k dataset:
1.  **LinUCB (Linear Upper Confidence Bound):** A deterministic algorithm optimizing for efficient exploration.
2.  **Linear Thompson Sampling (LinTS):** A probabilistic Bayesian approach for exploration.
3.  **Stochastic Reward Environment(SE-LinUCB) (Proposed):** A novel environment formulation that assigns probabilistic rewards to average ratings (3-stars) to maximize user engagement.

These online learning agents are benchmarked against a **Random Baseline** and an offline **Collaborative Filtering (SVD)** model to quantify the performance gap between cold-start (online) and warm-start (offline) scenarios.

## 📂 Project Structure
The codebase is organized as follows:

```
├── ml-100k/                # Dataset folder (Auto-downloaded on first run)
├── src/
│   ├── __init__.py
│   ├── environment.py      # Custom Gym Environments (Standard & Stochastic) + Data Loading
│   ├── agents.py           # LinUCB and Thompson Sampling Agent Classes
│   └── evaluation.py       # Metrics (Precision@K, Regret, SVD) and Plotting Functions
├── main.py                 # Main execution script (Orchestrator)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```


## ⚙️ Installation & Usage

### A. Prerequisites
Ensure you have Python installed along with the following libraries:  

```bash
pip install numpy pandas matplotlib scipy

```

### B. Dataset
Download Link [GroupLens](https://grouplens.org "MovieLens Dataset Source")  


### C. Running the Simulation
Execute the main script (or notebook) to run the 10,000-episode simulation:

```bash

python main.py
# Or run the Jupyter Notebook
jupyter notebook Bandit_Movie_Recommender.ipynb

```

### D. Experiment Flow (What happens when you run main.py):

1. **Data Loading:** Checks for ml-100k data; downloads and preprocesses it (User Context Vectors) if missing.

2. **Simulations:** Runs 10,000-episode simulations for:

    * Random Agent (Baseline)

    * LinUCB Agent

    * Linear Thompson Sampling Agent

    * Stochastic Reward Agent SE-LinUCB (Proposed Method)

3. **Evaluation:** Calculates Cumulative CTR, Regret, and Precision@5 metrics for all agents.  

4. **Benchmarking:** Trains an offline SVD model to establish the theoretical performance ceiling.  

5. **Visualization:** Generates and displays comparative learning curves and bar charts.


## 📊 Key Results:

* **Speed:** LinUCB achieved the fastest convergence for Click-Through Rate (CTR).  

* **Quality:** Linear Thompson Sampling demonstrated superior recommendation quality (Precision@5) compared to LinUCB.  

* **Engagement:** The proposed Stochastic Environment significantly increased total engagement volume (~53% CTR) by capturing passive user interest.


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