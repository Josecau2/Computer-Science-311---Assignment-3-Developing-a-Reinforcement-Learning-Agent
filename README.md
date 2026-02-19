# CartPole Reinforcement Learning Agent

**Computer Science 311 — Assignment 3: Developing a Reinforcement Learning Agent**

A reinforcement learning application that trains and compares two RL algorithms — **Q-Learning** (tabular) and **Deep Q-Network (DQN)** — on the Gymnasium CartPole-v1 environment. The agent learns to balance a pole on a moving cart through trial-and-error interactions.

## Overview

The CartPole problem requires an agent to decide whether to push a cart left or right at each timestep to keep a pole balanced upright. The agent receives +1 reward for each step the pole remains balanced, with episodes lasting up to 500 steps. This project implements two approaches:

- **Q-Learning:** Classical tabular method with discretized state space (20 bins per dimension)
- **DQN:** Deep Q-Network with experience replay buffer and target network (PyTorch)

## Results

| Metric | Q-Learning | DQN |
|--------|-----------|-----|
| Episodes | 5,000 | 500 |
| Mean Reward | 106.9 | 130.2 |
| Max Reward | 500 | 500 |
| Best 100-Ep Avg | 169.4 | 301.4 |

**DQN outperformed Q-Learning by 78%** in best 100-episode average, while using only 10% of the training episodes.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Josecau2/Computer-Science-311---Assignment-3-Developing-a-Reinforcement-Learning-Agent.git
cd Computer-Science-311---Assignment-3-Developing-a-Reinforcement-Learning-Agent

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the agent
python cartpole_rl_agent.py
```

## Output

The application generates 7 visualization plots in the `output/` directory:

| File | Description |
|------|-------------|
| `q_learning_training.png` | Q-Learning reward curve over 5,000 episodes |
| `dqn_training.png` | DQN reward curve over 500 episodes |
| `comparison_training.png` | Side-by-side training comparison |
| `dqn_loss.png` | DQN training loss curve |
| `epsilon_decay.png` | Exploration rate decay for both agents |
| `reward_distribution.png` | Histogram of episode rewards |
| `performance_summary.png` | Bar chart comparing key metrics |

## Requirements

- Python 3.10+
- PyTorch (CPU-only is sufficient)
- Gymnasium
- NumPy, Pandas, Matplotlib

See [requirements.txt](requirements.txt) for exact versions.

## Project Structure

```
├── cartpole_rl_agent.py    # Main application (Q-Learning + DQN)
├── requirements.txt        # Python dependencies
├── REPORT.md               # Detailed project report
├── README.md               # This file
└── output/                 # Generated plots (after running)
```

## Author

**Joseca Godoy** — Computer Science 311, Instructor: Daphane Olivar
