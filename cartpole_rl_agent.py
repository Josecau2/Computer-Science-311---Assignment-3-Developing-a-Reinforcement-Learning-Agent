"""
CartPole Reinforcement Learning Agent
Computer Science 311 — Assignment 3: Developing a Reinforcement Learning Agent
Instructor: Daphane Olivar
Author: Joseca Godoy
Date: February 2026

Trains an RL agent to balance a pole on a moving cart using the
Gymnasium CartPole-v1 environment. Implements and compares two algorithms:
1. Q-Learning (tabular, with discretized state space)
2. Deep Q-Network (DQN) using PyTorch

The agent learns optimal decisions through trial and error, improving
its performance over episodes of training.
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
RANDOM_SEED = 42
SOLVED_REWARD = 475  # CartPole-v1 is considered "solved" at 475+ avg reward

# Q-Learning hyperparameters
QL_EPISODES = 5000
QL_ALPHA = 0.1             # Learning rate
QL_GAMMA = 0.99            # Discount factor
QL_EPSILON_START = 1.0     # Initial exploration rate
QL_EPSILON_MIN = 0.01      # Minimum exploration rate
QL_EPSILON_DECAY = 0.9995  # Epsilon decay per episode

# DQN hyperparameters
DQN_EPISODES = 500
DQN_BATCH_SIZE = 64
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.01
DQN_EPSILON_DECAY = 0.995
DQN_LEARNING_RATE = 0.001
DQN_MEMORY_SIZE = 10000
DQN_TARGET_UPDATE = 10  # Update target network every N episodes
DQN_HIDDEN_SIZE = 128

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1: Q-LEARNING AGENT (Tabular)
# ══════════════════════════════════════════════════════════════════════════════


class QLearningAgent:
    """
    Q-Learning agent for CartPole with a discretized state space.

    The continuous 4D observation space is discretized into bins so that
    a standard Q-table can be used. The agent learns a mapping from
    (state, action) pairs to expected cumulative rewards.

    Q-Update Rule:
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
    """

    def __init__(self, env, n_bins=20):
        self.env = env
        self.n_bins = n_bins
        self.n_actions = env.action_space.n  # 2: push left or push right

        # Define discretization bins for each observation dimension
        # CartPole observations: [cart_pos, cart_vel, pole_angle, pole_vel]
        self.bins = [
            np.linspace(-4.8, 4.8, n_bins),       # Cart Position
            np.linspace(-4.0, 4.0, n_bins),        # Cart Velocity
            np.linspace(-0.418, 0.418, n_bins),    # Pole Angle (~24 degrees)
            np.linspace(-4.0, 4.0, n_bins),        # Pole Angular Velocity
        ]

        # Initialize Q-table with zeros
        # Shape: (n_bins+1, n_bins+1, n_bins+1, n_bins+1, n_actions)
        q_shape = tuple([n_bins + 1] * 4 + [self.n_actions])
        self.q_table = np.zeros(q_shape)

        # Hyperparameters
        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA
        self.epsilon = QL_EPSILON_START

    def discretize_state(self, observation: np.ndarray) -> tuple:
        """Convert continuous observation into discrete bin indices."""
        state = []
        for i, val in enumerate(observation):
            state.append(np.digitize(val, self.bins[i]))
        return tuple(state)

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection:
        - With probability epsilon: explore (random action)
        - With probability 1-epsilon: exploit (best Q-value action)
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return int(np.argmax(self.q_table[state]))  # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Apply the Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        """
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] = self.q_table[state + (action,)] + self.alpha * td_error

    def decay_epsilon(self):
        """Decay the exploration rate after each episode."""
        self.epsilon = max(QL_EPSILON_MIN, self.epsilon * QL_EPSILON_DECAY)


def train_q_learning():
    """Train the Q-Learning agent and return episode rewards."""
    print("\n" + "=" * 60)
    print("  TRAINING Q-LEARNING AGENT")
    print(f"  Episodes: {QL_EPISODES} | α: {QL_ALPHA} | γ: {QL_GAMMA}")
    print("=" * 60 + "\n")

    env = gym.make("CartPole-v1")
    agent = QLearningAgent(env)

    rewards_history = []
    avg_rewards = []
    best_avg = 0

    for episode in range(QL_EPISODES):
        observation, _ = env.reset(seed=RANDOM_SEED + episode)
        state = agent.discretize_state(observation)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize_state(next_obs)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        # Running average over last 100 episodes
        avg_100 = np.mean(rewards_history[-100:])
        avg_rewards.append(avg_100)

        if avg_100 > best_avg:
            best_avg = avg_100

        if (episode + 1) % 500 == 0:
            print(f"  Episode {episode + 1:>5}/{QL_EPISODES} | "
                  f"Reward: {total_reward:>6.0f} | "
                  f"Avg(100): {avg_100:>6.1f} | "
                  f"ε: {agent.epsilon:.4f}")

    env.close()
    print(f"\n  Training complete. Best Avg(100): {best_avg:.1f}")

    return rewards_history, avg_rewards


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2: DEEP Q-NETWORK (DQN) AGENT
# ══════════════════════════════════════════════════════════════════════════════


class DQNetwork(nn.Module):
    """
    Neural network that approximates the Q-value function.
    Takes a state as input and outputs Q-values for each action.

    Architecture: Input(4) → FC(128) → ReLU → FC(128) → ReLU → Output(2)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = DQN_HIDDEN_SIZE):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer — stores past transitions (s, a, r, s', done)
    and allows random sampling for mini-batch training.

    This breaks the temporal correlation between consecutive samples,
    stabilizing DQN training.
    """

    def __init__(self, capacity: int = DQN_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent using PyTorch.

    Key components:
    - Policy network: makes action decisions
    - Target network: provides stable Q-value targets (updated periodically)
    - Experience replay: stores transitions for mini-batch training
    - Epsilon-greedy exploration: balances exploration vs exploitation

    Loss function: MSE between predicted Q-values and target Q-values
    Target: r + γ max_a' Q_target(s', a')
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (trained every step)
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        # Target network (updated periodically for stability)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LEARNING_RATE)
        self.memory = ReplayBuffer(DQN_MEMORY_SIZE)
        self.epsilon = DQN_EPSILON_START

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection using the policy network."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a mini-batch from replay buffer and perform one gradient
        descent step on the policy network.

        Loss = MSE(Q_policy(s, a), r + γ max_a' Q_target(s', a'))
        """
        if len(self.memory) < DQN_BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(DQN_BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + DQN_GAMMA * next_q * (1 - dones_t)

        # Compute loss and backpropagate
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(DQN_EPSILON_MIN, self.epsilon * DQN_EPSILON_DECAY)


def train_dqn():
    """Train the DQN agent and return episode rewards and losses."""
    print("\n" + "=" * 60)
    print("  TRAINING DEEP Q-NETWORK (DQN) AGENT")
    print(f"  Episodes: {DQN_EPISODES} | Batch: {DQN_BATCH_SIZE} | γ: {DQN_GAMMA}")
    print(f"  Hidden: {DQN_HIDDEN_SIZE} | LR: {DQN_LEARNING_RATE}")
    print("=" * 60 + "\n")

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n             # 2

    agent = DQNAgent(state_size, action_size)
    print(f"  Device: {agent.device}")
    print(f"  Network: {agent.policy_net}\n")

    rewards_history = []
    avg_rewards = []
    losses = []
    best_avg = 0
    solved = False

    for episode in range(DQN_EPISODES):
        observation, _ = env.reset(seed=RANDOM_SEED + episode)
        state = observation
        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_obs, done)
            loss = agent.train_step()
            episode_loss += loss
            steps += 1

            state = next_obs
            total_reward += reward

        agent.decay_epsilon()

        # Update target network periodically
        if (episode + 1) % DQN_TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)
        avg_loss = episode_loss / max(steps, 1)
        losses.append(avg_loss)

        avg_100 = np.mean(rewards_history[-100:])
        avg_rewards.append(avg_100)

        if avg_100 > best_avg:
            best_avg = avg_100

        if avg_100 >= SOLVED_REWARD and not solved:
            solved = True
            print(f"  *** SOLVED at episode {episode + 1}! Avg(100): {avg_100:.1f} ***")

        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1:>4}/{DQN_EPISODES} | "
                  f"Reward: {total_reward:>6.0f} | "
                  f"Avg(100): {avg_100:>6.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.4f}")

    env.close()
    print(f"\n  Training complete. Best Avg(100): {best_avg:.1f}")
    if solved:
        print(f"  Environment SOLVED (Avg ≥ {SOLVED_REWARD})!")
    else:
        print(f"  Environment not fully solved (best: {best_avg:.1f}/{SOLVED_REWARD})")

    return rewards_history, avg_rewards, losses


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════


def plot_training_rewards(ql_rewards, ql_avg, dqn_rewards, dqn_avg, output_dir):
    """Plot training reward curves for both agents."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Q-Learning training curve ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ql_rewards, alpha=0.3, color="#3498db", label="Episode Reward")
    ax.plot(ql_avg, color="#e74c3c", linewidth=2, label="Avg (100 episodes)")
    ax.axhline(y=SOLVED_REWARD, color="green", linestyle="--", alpha=0.7, label=f"Solved ({SOLVED_REWARD})")
    ax.set_title("Q-Learning Agent — Training Progress", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "q_learning_training.png"), dpi=150)
    plt.close(fig)
    print("  Saved: q_learning_training.png")

    # ── DQN training curve ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dqn_rewards, alpha=0.3, color="#9b59b6", label="Episode Reward")
    ax.plot(dqn_avg, color="#e74c3c", linewidth=2, label="Avg (100 episodes)")
    ax.axhline(y=SOLVED_REWARD, color="green", linestyle="--", alpha=0.7, label=f"Solved ({SOLVED_REWARD})")
    ax.set_title("DQN Agent — Training Progress", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=150)
    plt.close(fig)
    print("  Saved: dqn_training.png")

    # ── Side-by-side comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(ql_avg, color="#3498db", linewidth=2)
    ax1.axhline(y=SOLVED_REWARD, color="green", linestyle="--", alpha=0.7)
    ax1.set_title("Q-Learning — Avg Reward", fontsize=12)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Reward (100 ep)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(dqn_avg, color="#9b59b6", linewidth=2)
    ax2.axhline(y=SOLVED_REWARD, color="green", linestyle="--", alpha=0.7)
    ax2.set_title("DQN — Avg Reward", fontsize=12)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Reward (100 ep)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Q-Learning vs DQN — Training Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_training.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_training.png")


def plot_dqn_loss(losses, output_dir):
    """Plot the DQN training loss curve."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(losses, alpha=0.4, color="#e67e22", label="Episode Avg Loss")
    # Smoothed loss
    if len(losses) >= 20:
        smoothed = pd.Series(losses).rolling(20).mean()
        ax.plot(smoothed, color="#c0392b", linewidth=2, label="Smoothed (20 ep)")
    ax.set_title("DQN Agent — Training Loss", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "dqn_loss.png"), dpi=150)
    plt.close(fig)
    print("  Saved: dqn_loss.png")


def plot_epsilon_decay(output_dir):
    """Visualize the epsilon decay schedules for both agents."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Q-Learning epsilon decay
    eps_ql = [QL_EPSILON_START]
    for _ in range(QL_EPISODES - 1):
        eps_ql.append(max(QL_EPSILON_MIN, eps_ql[-1] * QL_EPSILON_DECAY))
    ax1.plot(eps_ql, color="#3498db", linewidth=1.5)
    ax1.set_title("Q-Learning — Epsilon Decay", fontsize=12)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Epsilon (ε)")
    ax1.grid(True, alpha=0.3)

    # DQN epsilon decay
    eps_dqn = [DQN_EPSILON_START]
    for _ in range(DQN_EPISODES - 1):
        eps_dqn.append(max(DQN_EPSILON_MIN, eps_dqn[-1] * DQN_EPSILON_DECAY))
    ax2.plot(eps_dqn, color="#9b59b6", linewidth=1.5)
    ax2.set_title("DQN — Epsilon Decay", fontsize=12)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon (ε)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Exploration Rate Decay Over Training", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "epsilon_decay.png"), dpi=150)
    plt.close(fig)
    print("  Saved: epsilon_decay.png")


def plot_reward_distribution(ql_rewards, dqn_rewards, output_dir):
    """Plot histogram of episode rewards for both agents."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(ql_rewards, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    ax1.axvline(np.mean(ql_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(ql_rewards):.1f}")
    ax1.set_title("Q-Learning — Reward Distribution", fontsize=12)
    ax1.set_xlabel("Episode Reward")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(dqn_rewards, bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
    ax2.axvline(np.mean(dqn_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(dqn_rewards):.1f}")
    ax2.set_title("DQN — Reward Distribution", fontsize=12)
    ax2.set_xlabel("Episode Reward")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Episode Reward Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: reward_distribution.png")


def plot_performance_summary(ql_rewards, dqn_rewards, output_dir):
    """Bar chart comparing key metrics between both agents."""
    metrics = {
        "Mean Reward": [np.mean(ql_rewards), np.mean(dqn_rewards)],
        "Max Reward": [np.max(ql_rewards), np.max(dqn_rewards)],
        "Std Dev": [np.std(ql_rewards), np.std(dqn_rewards)],
        "Last 100 Avg": [np.mean(ql_rewards[-100:]), np.mean(dqn_rewards[-100:])],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#3498db", "#9b59b6"]
    labels = ["Q-Learning", "DQN"]

    for idx, (metric, values) in enumerate(metrics.items()):
        bars = axes[idx].bar(labels, values, color=colors, edgecolor="black", alpha=0.8)
        axes[idx].set_title(metric, fontsize=12)
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f"{val:.1f}", ha="center", va="bottom", fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Q-Learning vs DQN — Performance Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "performance_summary.png"), dpi=150)
    plt.close(fig)
    print("  Saved: performance_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def print_results(ql_rewards, dqn_rewards):
    """Print formatted comparison of both agents."""
    print("\n" + "=" * 70)
    print("  CARTPOLE RL AGENT — FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Q-Learning':>15} {'DQN':>15}")
    print("  " + "-" * 57)

    metrics = [
        ("Total Episodes", f"{len(ql_rewards)}", f"{len(dqn_rewards)}"),
        ("Mean Reward", f"{np.mean(ql_rewards):.1f}", f"{np.mean(dqn_rewards):.1f}"),
        ("Max Reward", f"{np.max(ql_rewards):.0f}", f"{np.max(dqn_rewards):.0f}"),
        ("Min Reward", f"{np.min(ql_rewards):.0f}", f"{np.min(dqn_rewards):.0f}"),
        ("Std Deviation", f"{np.std(ql_rewards):.1f}", f"{np.std(dqn_rewards):.1f}"),
        ("Last 100 Avg", f"{np.mean(ql_rewards[-100:]):.1f}", f"{np.mean(dqn_rewards[-100:]):.1f}"),
        ("Best 100-Ep Avg", f"{max(pd.Series(ql_rewards).rolling(100).mean().dropna()):.1f}",
                            f"{max(pd.Series(dqn_rewards).rolling(100).mean().dropna()):.1f}"),
    ]

    for name, ql_val, dqn_val in metrics:
        print(f"  {name:<25} {ql_val:>15} {dqn_val:>15}")

    print("=" * 70)

    # Determine winner
    ql_best = np.mean(ql_rewards[-100:])
    dqn_best = np.mean(dqn_rewards[-100:])
    winner = "DQN" if dqn_best > ql_best else "Q-Learning"
    print(f"\n  Best agent (last 100 avg): {winner}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("\n" + "=" * 70)
    print("  CARTPOLE REINFORCEMENT LEARNING AGENT")
    print("  Q-Learning vs Deep Q-Network (DQN)")
    print("  Environment: Gymnasium CartPole-v1")
    print("=" * 70)

    # ── Describe the environment ──
    env = gym.make("CartPole-v1")
    print(f"\n  Environment: CartPole-v1")
    print(f"  Observation space: {env.observation_space} (4 continuous values)")
    print(f"  Action space: {env.action_space} (2 discrete: left/right)")
    print(f"  Max steps per episode: 500")
    print(f"  Solved threshold: Avg reward ≥ {SOLVED_REWARD} over 100 episodes")
    env.close()

    # ── Train Q-Learning Agent ──
    ql_rewards, ql_avg = train_q_learning()

    # ── Train DQN Agent ──
    dqn_rewards, dqn_avg, dqn_losses = train_dqn()

    # ── Print results ──
    print_results(ql_rewards, dqn_rewards)

    # ── Generate all visualizations ──
    print("\nGenerating visualizations ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_training_rewards(ql_rewards, ql_avg, dqn_rewards, dqn_avg, OUTPUT_DIR)
    plot_dqn_loss(dqn_losses, OUTPUT_DIR)
    plot_epsilon_decay(OUTPUT_DIR)
    plot_reward_distribution(ql_rewards, dqn_rewards, OUTPUT_DIR)
    plot_performance_summary(ql_rewards, dqn_rewards, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nApplication complete.")


if __name__ == "__main__":
    main()
