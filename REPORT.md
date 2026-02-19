# Developing a Reinforcement Learning Agent

**Course:** Computer Science 311 — Assignment 3: Developing a Reinforcement Learning Agent  
**Instructor:** Daphane Olivar  
**Author:** Joseca Godoy  
**Date:** February 2026

---

## 1. Name and Purpose of the Agent

**Name:** CartPole RL Agent

The CartPole RL Agent is a reinforcement learning system designed to learn the task of balancing a pole on a moving cart through trial and error. The agent interacts with the Gymnasium CartPole-v1 simulated environment, where it must decide at each timestep whether to push the cart left or right to prevent a pole — hinged to the top of the cart — from falling over. The application implements and compares two RL algorithms: tabular Q-Learning (with a discretized state space) and a Deep Q-Network (DQN) using PyTorch. The agent receives a reward of +1 for each timestep the pole remains balanced, learning over thousands of episodes to maximize cumulative reward. The CartPole-v1 environment is considered "solved" when the agent achieves an average reward of 475 or higher over 100 consecutive episodes, meaning the pole stays balanced for nearly the entire 500-step episode limit.

---

## 2. Algorithms Used

Two reinforcement learning algorithms were implemented and compared:

### 2.1 Q-Learning (Tabular)

Q-Learning is a model-free, off-policy temporal difference (TD) RL algorithm that learns a state-action value function $Q(s, a)$ representing the expected cumulative discounted reward of taking action $a$ in state $s$ and following the optimal policy thereafter (Watkins & Dayan, 1992). The Q-value is updated after each step using the TD update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha = 0.1$ is the learning rate
- $\gamma = 0.99$ is the discount factor
- $r$ is the immediate reward
- $s'$ is the next state

**Why chosen:** Q-Learning is the foundational RL algorithm and provides a clear baseline for comparison. It demonstrates core RL concepts (states, actions, rewards, exploration-exploitation tradeoff) in their simplest form. However, because CartPole has a continuous state space, the observations must be discretized into bins — a limitation that motivates the DQN approach.

**Exploration strategy:** Epsilon-greedy with $\epsilon$ decaying from 1.0 to 0.01 over training (decay rate: 0.9995 per episode). This ensures the agent explores broadly early in training and increasingly exploits its learned policy over time.

### 2.2 Deep Q-Network (DQN)

DQN extends Q-Learning by replacing the Q-table with a neural network that approximates $Q(s, a)$ for continuous state spaces (Mnih et al., 2015). The neural network takes the raw 4-dimensional state vector as input and outputs Q-values for each action.

**Key innovations used:**

| Component | Description |
|-----------|-------------|
| **Experience Replay** | Transitions $(s, a, r, s', done)$ are stored in a replay buffer of size 10,000. Mini-batches of 64 transitions are sampled randomly for training, breaking temporal correlations that destabilize learning. |
| **Target Network** | A separate target network provides stable Q-value targets during training. Its weights are copied from the policy network every 10 episodes, preventing the "moving target" problem where the network chases its own changing predictions. |
| **Gradient Clipping** | Gradients are clipped to a maximum norm of 1.0 to prevent exploding gradients and stabilize training. |

**Network architecture:** Input(4) → FC(128) → ReLU → FC(128) → ReLU → Output(2)

**Loss function:** Mean Squared Error between predicted Q-values and target Q-values:

$$L = \frac{1}{N} \sum_{i} \left( Q_{\text{policy}}(s_i, a_i) - \left[ r_i + \gamma \max_{a'} Q_{\text{target}}(s_i', a') \right] \right)^2$$

**Why chosen:** DQN solves Q-Learning's fundamental limitation for continuous state spaces by using function approximation. It is the algorithm that demonstrated superhuman performance on Atari games (Mnih et al., 2015) and remains one of the most widely used deep RL methods. It naturally handles the 4-dimensional continuous observation space of CartPole without discretization.

---

## 3. Dataset Information

### 3.1 Dataset Source

**Environment:** Gymnasium CartPole-v1  
**Source:** Farama Foundation — https://gymnasium.farama.org/environments/classic_control/cart_pole/  
**Original:** Based on Barto et al. (1983), "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems."

Unlike supervised learning, reinforcement learning does not use a static dataset. Instead, data is generated dynamically through agent-environment interactions. Each episode produces a trajectory of state-action-reward transitions. The "dataset" grows during training.

### 3.2 Number of Records

| Metric | Q-Learning | DQN |
|--------|-----------|-----|
| Training episodes | 5,000 | 500 |
| Transitions per episode | Up to 500 | Up to 500 |
| Total transitions generated | ~500,000+ | ~100,000+ |
| Replay buffer size (DQN) | — | 10,000 (max) |

### 3.3 Number of Features

- **Observation space:** 4 continuous features (state variables)
- **Action space:** 2 discrete actions (push left or push right)

### 3.4 Description of Features

**Observation Space (State):**

| Feature Name | Description | Data Type | Range |
|--------------|-------------|-----------|-------|
| Cart Position | Horizontal position of the cart on the track | Float (continuous) | [-4.8, 4.8] |
| Cart Velocity | Horizontal velocity of the cart | Float (continuous) | [-∞, ∞] |
| Pole Angle | Angle of the pole from vertical (radians) | Float (continuous) | [-0.418, 0.418] (~±24°) |
| Pole Angular Velocity | Rate of change of the pole angle | Float (continuous) | [-∞, ∞] |

**Action Space:**

| Action | Description |
|--------|-------------|
| 0 | Push cart to the **left** |
| 1 | Push cart to the **right** |

**Reward Signal:**

| Condition | Reward |
|-----------|--------|
| Each timestep the pole is balanced | +1 |
| Episode termination (pole falls or cart out of bounds) | 0 (episode ends) |
| Maximum episode length | 500 steps |

**Termination Conditions:**
- Pole angle exceeds ±12° from vertical
- Cart position exceeds ±2.4 (falls off track)
- Episode length reaches 500 steps (success)

### 3.5 Preprocessing Steps

**Q-Learning Discretization:**
1. Each of the 4 continuous observation dimensions was divided into 20 equal-width bins.
2. Cart Position bins: [-4.8, 4.8], Cart Velocity bins: [-4.0, 4.0], Pole Angle bins: [-0.418, 0.418], Pole Angular Velocity bins: [-4.0, 4.0].
3. Q-table shape: (21, 21, 21, 21, 2) = 388,962 state-action entries.

**DQN Input Processing:**
1. Raw continuous observations are fed directly to the neural network — no discretization needed.
2. The network handles the continuous-to-discrete mapping internally through learned representations.

---

## 4. Libraries, Toolkits, and Frameworks

| Library / Toolkit | Version | Role |
|-------------------|---------|------|
| **Gymnasium** | ≥ 0.29.0 | RL environment framework by the Farama Foundation. Provides the CartPole-v1 simulation with standardized step/reset API for agent-environment interaction. |
| **PyTorch** | ≥ 2.0.0 | Deep learning framework used to build and train the DQN neural network. Provides `nn.Module` for network definition, `optim.Adam` for optimization, and autograd for backpropagation. |
| **NumPy** | ≥ 1.24.0 | Numerical computing library. Used for array operations, Q-table initialization, state discretization (`np.digitize`), and reward statistics. |
| **Pandas** | ≥ 2.0.0 | Data manipulation library. Used for computing rolling averages of training rewards and formatting results. |
| **Matplotlib** | ≥ 3.7.0 | Visualization library. Generates all training curves, reward distributions, epsilon decay plots, loss curves, and performance comparison charts. |
| **collections.deque** | stdlib | Used for the Experience Replay Buffer — fixed-size queue that automatically discards oldest transitions when capacity is reached. |

---

## 5. Application Design and Implementation

### Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                  CartPole-v1 Environment                   │
│  Observation: [cart_pos, cart_vel, pole_angle, pole_vel]   │
│  Actions: [push_left, push_right]                          │
│  Reward: +1 per balanced timestep                          │
└────────────────────┬──────────────────────────────────────┘
                     │ observation, reward, done
                     ▼
         ┌──────────────────────┐
         │   Agent Selection     │
         └──────┬───────┬───────┘
                │       │
    ┌───────────▼──┐  ┌─▼────────────────┐
    │  Q-Learning   │  │  DQN Agent        │
    │  ┌──────────┐ │  │  ┌─────────────┐ │
    │  │ Discretize│ │  │  │ Policy Net  │ │
    │  │ State     │ │  │  │ (128-128-2) │ │
    │  └────┬─────┘ │  │  └──────┬──────┘ │
    │       ▼       │  │         ▼         │
    │  ┌──────────┐ │  │  ┌─────────────┐ │
    │  │ Q-Table   │ │  │  │ Target Net  │ │
    │  │ (21⁴ × 2)│ │  │  │ (stable Q)  │ │
    │  └────┬─────┘ │  │  └──────┬──────┘ │
    │       ▼       │  │         ▼         │
    │  ε-greedy     │  │  ┌─────────────┐ │
    │  action       │  │  │ Replay      │ │
    │               │  │  │ Buffer      │ │
    │               │  │  │ (10,000)    │ │
    │               │  │  └─────────────┘ │
    └───────────────┘  └──────────────────┘
                │               │
                ▼               ▼
        ┌──────────────────────────┐
        │  Evaluation & Comparison  │
        │  Training Plots           │
        │  Performance Metrics      │
        └──────────────────────────┘
```

### Training Loop (for both agents)

1. **Reset:** Environment resets; agent receives initial observation.
2. **Action Selection:** Agent uses ε-greedy policy — with probability ε selects a random action (exploration), otherwise selects the action with highest Q-value (exploitation).
3. **Step:** Environment executes the action, returns next state, reward, and done flag.
4. **Learning:**
   - *Q-Learning:* Updates the Q-table entry for (state, action) using the TD update rule.
   - *DQN:* Stores the transition in the replay buffer, samples a mini-batch, computes loss against target network, and performs a gradient descent step.
5. **Repeat:** Steps 2–4 continue until the episode terminates.
6. **Epsilon Decay:** After each episode, ε is decayed to shift the agent from exploration toward exploitation.
7. **Target Update (DQN only):** Every 10 episodes, the target network weights are synced with the policy network.

### Key Design Decisions

- **Two-Algorithm Comparison:** Implementing both Q-Learning and DQN demonstrates the progression from tabular to function-approximation methods and provides a fair comparison of classical vs. deep RL.
- **Epsilon-Greedy Exploration:** Both agents use ε-greedy with exponential decay, ensuring broad exploration early and refined exploitation later.
- **Experience Replay (DQN):** Random sampling from the replay buffer breaks temporal correlation between consecutive training samples, which is critical for stable neural network training.
- **Target Network (DQN):** Periodic weight synchronization (every 10 episodes) provides stable learning targets, preventing the oscillation and divergence common when using a single network for both prediction and target computation.

---

## 6. Instructions for Running the Agent

### Prerequisites

- Python 3.10 or higher
- No GPU required (CPU-only training completes in a few minutes)

### Step-by-Step Guide

```bash
# 1. Clone the repository
git clone https://github.com/Josecau2/Computer-Science-311---Assignment-3-Developing-a-Reinforcement-Learning-Agent.git
cd Computer-Science-311---Assignment-3-Developing-a-Reinforcement-Learning-Agent

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the RL agent
python cartpole_rl_agent.py
```

### Expected Output

The application will:
1. Display environment information (observation space, action space)
2. Train the Q-Learning agent (5,000 episodes, progress every 500 episodes)
3. Train the DQN agent (500 episodes, progress every 50 episodes)
4. Print a comparison table of both agents' performance
5. Save 7 visualization plots to the `output/` directory

### Output Files

After running, the `output/` directory will contain:
- `q_learning_training.png` — Q-Learning reward curve over episodes
- `dqn_training.png` — DQN reward curve over episodes
- `comparison_training.png` — Side-by-side training comparison
- `dqn_loss.png` — DQN training loss curve
- `epsilon_decay.png` — Exploration rate decay for both agents
- `reward_distribution.png` — Histogram of episode rewards
- `performance_summary.png` — Bar chart comparing key metrics

---

## 7. Results

### Performance Summary

| Metric | Q-Learning | DQN |
|--------|-----------|-----|
| Total Episodes | 5,000 | 500 |
| Mean Reward | 106.9 | 130.2 |
| Max Reward | 500 | 500 |
| Min Reward | 8 | 8 |
| Std Deviation | 69.3 | 135.4 |
| Last 100-Episode Avg | 152.8 | 227.4 |
| Best 100-Episode Avg | 169.4 | 301.4 |
| Final Epsilon | 0.0820 | 0.0816 |
| Solved (≥475 avg) | No | No (best: 301.4) |

**Winner: DQN** — achieved a best 100-episode average of 301.4 compared to Q-Learning's 169.4, representing a 78% improvement. While neither agent fully solved the environment (≥475 threshold), DQN demonstrated substantially stronger learning capacity with 10× fewer episodes.

### Key Observations

1. **DQN significantly outperforms Q-Learning** on CartPole despite training for only 500 episodes vs. 5,000. The DQN's best 100-episode average (301.4) was 78% higher than Q-Learning's (169.4). The continuous state space loses critical information during discretization, limiting Q-Learning's ability to learn fine-grained control, while DQN processes the raw continuous observations and learns a much more precise policy.

2. **DQN learns more efficiently** — with only 10% of the training episodes, DQN achieved nearly double the average performance. The experience replay and target network stabilize learning, while the neural network's representational capacity allows it to generalize across similar states.

3. **Both agents reached the maximum reward of 500** in individual episodes, demonstrating that both algorithms can discover the optimal balancing strategy. However, Q-Learning was unable to sustain this performance consistently, with high variance across episodes (std: 69.3) compared to DQN's higher variance (std: 135.4) which reflects the agent's transition from exploration to exploitation.

4. **Epsilon decay** is essential for both agents. Both agents' final epsilon values converged to similar levels (~0.08), indicating comparable exploration-exploitation schedules. Early episodes show low rewards as the agent explores randomly, followed by improvement as exploitation increases.

5. **DQN's training loss** generally decreases over time but exhibits variance due to the non-stationary nature of RL training (the target distribution shifts as the policy improves).

### Visualizations

All 7 plots are saved in the `output/` directory, including training reward curves, loss curves, epsilon decay schedules, reward distributions, and a performance summary comparison.

---

## 8. Discussion and Insights

### Performance Analysis

The DQN agent demonstrated clear superiority over the tabular Q-Learning agent on the CartPole task. This result aligns with the theoretical expectation: Q-Learning requires discretization of the continuous state space, which inherently loses information. Small differences in cart velocity or pole angle — which are crucial for fine-grained balance control — are collapsed into the same discrete bin, preventing the agent from distinguishing between them.

The DQN agent, by contrast, processes the raw continuous observations through a neural network that can learn arbitrarily precise decision boundaries. The combination of experience replay and a target network — the two key innovations from Mnih et al. (2015) — proved essential for stable training. Without experience replay, consecutive correlated samples would destabilize the network weights. Without the target network, the Q-value targets would shift with each weight update, creating a "moving target" that prevents convergence.

### Limitations

1. **Q-Learning Discretization:** The 20-bin discretization is a coarse approximation. Increasing bins improves precision but exponentially inflates the Q-table size (curse of dimensionality). With 4 dimensions and 20 bins each, the Q-table already has ~390,000 entries.

2. **Sample Efficiency:** Both algorithms require thousands of episodes of trial-and-error interactions. In real-world applications (e.g., robotics), this level of exploration may be impractical or dangerous without simulation environments.

3. **Single Environment:** The agents are trained and evaluated on CartPole only. Transfer to other tasks (e.g., different physics, observation spaces) would require retraining from scratch.

4. **Hyperparameter Sensitivity:** Both algorithms' performance is sensitive to hyperparameter choices (learning rate, epsilon decay rate, network architecture). No systematic hyperparameter search was conducted.

5. **No Prioritized Replay:** The DQN uses uniform random sampling from the replay buffer. Prioritized experience replay, which samples transitions with higher TD-error more frequently, could improve training efficiency (Schaul et al., 2016).

### Potential Improvements

1. **Double DQN:** Use the policy network to select actions and the target network to evaluate them, reducing the overestimation bias present in standard DQN (van Hasselt et al., 2016).

2. **Dueling DQN:** Separate the Q-value estimation into state value and action advantage streams, allowing the network to learn which states are valuable regardless of the action taken (Wang et al., 2016).

3. **Prioritized Experience Replay:** Sample transitions with higher learning potential more frequently, improving sample efficiency (Schaul et al., 2016).

4. **Policy Gradient Methods:** Algorithms like PPO (Proximal Policy Optimization) or A3C (Asynchronous Advantage Actor-Critic) learn a stochastic policy directly, which can be more stable and effective for continuous control tasks (Schulman et al., 2017).

5. **Hyperparameter Tuning:** Apply grid search or Bayesian optimization over learning rate, network architecture, batch size, and epsilon decay to find optimal configurations.

### Real-World Applicability

The CartPole balance problem is a simplified analogue of many real-world control tasks: self-balancing robots, rocket landing stabilization, and industrial process control all involve making continuous adjustments to prevent system failure. The progression from tabular Q-Learning to DQN demonstrated in this project mirrors the historical progression of RL from theoretical toy problems to practical applications in robotics, autonomous driving, and game AI (Silver et al., 2016). The core concepts of exploration-exploitation tradeoff, temporal difference learning, and experience replay are foundational to all modern RL systems.

---

## 9. References

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics, SMC-13*(5), 834–846. https://doi.org/10.1109/TSMC.1983.6313077

Farama Foundation. (2024). *Gymnasium: A standard API for reinforcement learning*. https://gymnasium.farama.org/

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*, 8026–8037. https://pytorch.org/

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *Proceedings of the International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1511.05952

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*. https://arxiv.org/abs/1707.06347

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature, 529*(7587), 484–489. https://doi.org/10.1038/nature16961

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book-2nd.html

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1), 2094–2100. https://doi.org/10.1609/aaai.v30i1.10295

Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *Proceedings of the International Conference on Machine Learning (ICML)*, 1995–2003. https://arxiv.org/abs/1511.06581

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3), 279–292. https://doi.org/10.1007/BF00992698
