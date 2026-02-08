# RLML  
## Reinforcement Learning-based Meta-Learner for Classification

RLML (Reinforcement Learning-based Meta-Learner) is an advanced meta-learning framework that formulates the meta-learner construction problem as a Reinforcement Learning (RL) task.  
The algorithm dynamically selects base learners and feature subsets by interacting with an environment, optimizing classification performance through learned policies rather than fixed heuristics.

This repository provides the **final reference implementation** of the RLML algorithm.

---

## Key Features

- Reinforcement Learning-driven meta-learning architecture  
- Dynamic base-learner and feature subset selection  
- Environment–agent formulation for meta-learning  
- Curriculum Learning support for faster convergence  
- Automated training, validation, and testing cycles  
- Designed for high-dimensional and large-scale datasets  

---

## Algorithm Overview

RLML models the meta-learning process as a Reinforcement Learning problem:

- **State:** Current feature subset, base-learner configuration, and performance signals  
- **Action:** Selection or elimination of base learners and/or features  
- **Reward:** Performance-based feedback (e.g., accuracy, F1-score improvements)  
- **Policy:** Learned strategy that optimizes long-term classification performance  

Through iterative interaction with the environment, the agent learns an optimal policy for constructing a robust meta-learner.

---

## Repository Structure

```
├── data/                     # Partial sample datasets (for structure only)
├── globals.py                # Global configuration and environment settings
├── rlml_environment.py       # RL environment definition
├── rlml_agent.py             # RL agent (Q-learning / DQN-based)
├── train_rlml.py             # Training script
├── evaluate_rlml.py          # Evaluation and testing script
├── base_learners/            # Base learning algorithms
├── utils/                    # Helper functions and utilities
└── README.md
```

**Important:**  
The `data/` directory contains **partial samples only**.  
Full datasets must be obtained from their original sources (e.g., UCLA, UCI, Kaggle).

---

## Setup & Configuration

### 1. Configure Global Settings

Edit `globals.py` and ensure:
- Dataset paths are correctly set  
- Output and logging directories exist  
- RL-related parameters (episodes, steps, rewards) are defined  

This step is mandatory before running RLML.

---

### 2. Enable Base Learners

In `globals.py`, explicitly enable the base classifiers to be used by the RL agent, such as:
- Random Forest (RF)  
- XGBoost (XGB)  
- Support Vector Machines (SVM)  

Only enabled learners will be considered during policy learning.

---

### 3. Hyperparameters

Common RL-related hyperparameters include:
- Learning rate (α)  
- Discount factor (γ)  
- Exploration rate (ε)  
- Number of episodes  
- Maximum steps per episode  

These parameters can be tuned in `globals.py` or related configuration files.

---

## Training RLML

To train the RL-based meta-learner, run:

```bash
python train_rlml.py
```

During training:
- The agent interacts with the environment  
- Rewards are collected after each action  
- Policies are updated incrementally  

Curriculum Learning can be enabled to speed up convergence by starting with smaller data subsets.

---

## Evaluation & Testing

After training, evaluate the learned policy using:

```bash
python evaluate_rlml.py
```

This script:
- Loads the trained policy  
- Applies it to unseen data  
- Reports classification performance metrics  

---

## Example Workflow

1. Configure paths and learners in `globals.py`  
2. Adjust RL hyperparameters  
3. Train the RLML agent  
4. Evaluate the trained meta-learner  

---

## Datasets

RLML has been tested on multiple benchmark datasets.  
Due to licensing and size constraints, full datasets are **not included**.

Please download datasets from their original sources (e.g., UCLA, UCI, Kaggle) and place them under the `data/` directory.

---

## Reference

If you use this code in academic or industrial work, please cite the relevant RLML publication(s).

(Reference details to be added or updated here if applicable)

---

## License

This project is provided for **research and educational purposes**.  
Please consult the repository license file for detailed usage terms.

---

## Contributions

Contributions, bug reports, and extensions are welcome.  
For significant changes, please open an issue first to discuss your proposal.
