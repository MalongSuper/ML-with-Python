# Reinforcement Learning - The Maze Game
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Helper Functions: Check if the move is valid
def is_valid(pos):
    r, c = pos
    if r < 0 or r >= maze.shape[0]:
        return False
    if c < 0 or c >= maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True


def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))  # explore
    else:
        return np.argmax(Q[state])  # exploit


def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break

        visited.add(state)
        best_action = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])

            if (0 <= next_state[0] < maze.shape[0] and
                0 <= next_state[1] < maze.shape[1] and
                maze[next_state] == 0 and
                next_state not in visited):

                if Q[state][idx] > best_value:
                    best_value = Q[state][idx]
                    best_action = idx

        if best_action is None:
            break

        move = actions[best_action]
        state = (state[0] + move[0], state[1] + move[1])
        path.append(state)

    return path

def plot_maze_with_path(path):
    cmap = ListedColormap(['#eef8ea', '#a8c79c'])
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    plt.scatter(start[1], start[0], marker='o', color='#81c784',
                edgecolors='black', s=200, label='Start', zorder=5)

    plt.scatter(goal[1], goal[0], marker='*', color='#388e3c',
                edgecolors='black', s=300, label='Goal', zorder=5)

    rows, cols = zip(*path)
    plt.plot(cols, rows, color='#60b37a', linewidth=4,
             label='Learned Path', zorder=4)

    plt.title('Reinforcement Learning: Maze Navigation')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # explore
    else:
        return np.argmax(Q[state[0], state[1]])  # exploit


def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, 4))  # 4 actions

    local_eps = epsilon
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, local_eps)

        done = False

        while not done:
            next_state, reward, done = env.step(action)

            if not done:
                next_action = epsilon_greedy_policy(Q, next_state, local_eps)
                target = reward + gamma * Q[next_state[0], next_state[1], next_action]
            else:
                next_action = None
                target = reward

            Q[state[0], state[1], action] += alpha * (target - Q[state[0], state[1], action])

            state = next_state
            # if terminal, stop using next_action
            if next_action is None:
                break
            action = next_action

        local_eps = max(0.01, local_eps * 0.995)

    return Q




def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, 4))  # 4 actions

    local_eps = epsilon
    for episode in range(episodes):
        state = env.reset()

        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, local_eps)
            next_state, reward, done = env.step(action)

            best_next = np.max(Q[next_state[0], next_state[1]]) if not done else 0

            Q[state[0], state[1], action] += alpha * (reward + gamma * best_next
                                                      - Q[state[0], state[1], action])

            state = next_state

        local_eps = max(0.01, local_eps * 0.995)

    return Q

maze = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                 [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
                 [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
                 [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
                 [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]])

start = (0, 0)
goal = (9, 9)

actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down

num_episodes = 5000
alpha = 0.1      # learning rate
gamma = 0.9      # future importance
epsilon = 0.5    # exploration

reward_fire = -10
reward_goal = 50
reward_step = -1


class MazeEnv:
    def __init__(self, maze, start, goal, actions,
                 reward_fire=-10, reward_goal=50, reward_step=-1):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.actions = actions
        self.height, self.width = maze.shape
        self.reward_fire = reward_fire
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def is_valid(self, pos):
        r, c = pos
        if r < 0 or r >= self.height:
            return False
        if c < 0 or c >= self.width:
            return False
        if self.maze[r, c] == 1:
            return False
        return True

    def step(self, action_index):
        move = self.actions[action_index]
        next_state = (self.state[0] + move[0], self.state[1] + move[1])

        if not self.is_valid(next_state):
            reward = self.reward_fire
            done = True
        elif next_state == self.goal:
            reward = self.reward_goal
            done = True
            self.state = next_state
        else:
            reward = self.reward_step
            done = False
            self.state = next_state

        return self.state, reward, done


def episodic_q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, len(env.actions)))

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            next_max = np.max(Q[next_state[0], next_state[1]]) if not done else 0

            Q[state[0], state[1], action] += alpha * (reward + gamma * next_max
                                                       - Q[state[0], state[1], action])

            state = next_state

        epsilon = max(0.01, epsilon * 0.995)

    return Q


def plot_maze_with_paths(paths, labels, colors=None):
    cmap = ListedColormap(['#eef8ea', '#a8c79c'])
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    plt.scatter(start[1], start[0], marker='o', color='lightgreen',
                edgecolors='black', s=200, label='Start', zorder=5)

    plt.scatter(goal[1], goal[0], marker='*', color='green',
                edgecolors='black', s=300, label='Goal', zorder=5)

    if colors is None:
        colors = ['red', 'blue', 'green']

    for path, label, color in zip(paths, labels, colors):
        if not path:
            continue
        rows, cols = zip(*path)
        plt.plot(cols, rows, color=color, linewidth=4,
                 label=label, zorder=4)

    plt.title('Reinforcement Learning: Maze Navigation (3 algorithms)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def plot_three_maze_subplots(paths, labels, colors=None):
    if colors is None:
        colors = ['red', 'blue', 'green']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, path, label, color in zip(axes, paths, labels, colors):
        cmap = ListedColormap(['#eef8ea', '#a8c79c'])
        ax.imshow(maze, cmap=cmap)

        ax.scatter(start[1], start[0], marker='o', color='lightgreen',
               edgecolors='black', s=150, zorder=5)
        ax.scatter(goal[1], goal[0], marker='*', color='green',
               edgecolors='black', s=200, zorder=5)

        if path:
            rows, cols = zip(*path)
            ax.plot(cols, rows, color=color, linewidth=4, zorder=4)

        ax.set_title(label)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_qtable_subplots(Qs, labels, cmap='viridis'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    h, w = maze.shape
    for ax, Q, label in zip(axes, Qs, labels):
        Qmax = np.max(Q, axis=2)
        im = ax.imshow(Qmax, cmap=cmap)
        ax.set_title(label + ' (max action value)')
        ax.invert_yaxis()
        ax.grid(False)

        # annotate values and mark walls
        for i in range(h):
            for j in range(w):
                if maze[i, j] == 1:
                    ax.text(j, i, 'X', ha='center', va='center', color='white', fontsize=8)
                else:
                    ax.text(j, i, f"{Qmax[i, j]:.1f}", ha='center', va='center', color='white', fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = MazeEnv(maze, start, goal, actions,
                  reward_fire=reward_fire, reward_goal=reward_goal, reward_step=reward_step)

    # Run SARSA
    Q_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)

    # Run Q-Learning (function defined above)
    Q_qlearn = q_learning(env, num_episodes, alpha, gamma, epsilon)

    # Run episodic tabular Q-learning (same style as original loop)
    Q_episodic = episodic_q_learning(env, num_episodes, alpha, gamma, epsilon)

    # Extract optimal paths
    path_sarsa = get_optimal_path(Q_sarsa, start, goal, actions, maze)
    path_qlearn = get_optimal_path(Q_qlearn, start, goal, actions, maze)
    path_episodic = get_optimal_path(Q_episodic, start, goal, actions, maze)

    labels = ['SARSA', 'Q-Learning (Async)', 'Q-Learning (Episodic)']

    # Plot three maze subplots (one per algorithm)
    plot_three_maze_subplots([path_sarsa, path_qlearn, path_episodic], labels)

    # Plot Q-table heatmaps (max over actions) for each algorithm
    plot_qtable_subplots([Q_sarsa, Q_qlearn, Q_episodic], labels)

