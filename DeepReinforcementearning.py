# DeepRL_Graph_Friend_Unique.py

import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------- ENVIRONMENT ----------------
edges = [
    (0,1),(1,5),(5,6),(5,4),(1,2),
    (1,3),(9,10),(2,4),(0,6),(6,7),
    (7,8),(8,9),(1,7),(3,9)
]

N_STATES = 11
GOAL = 10

# ---------------- REWARD MATRIX (CHANGED) ----------------
R = np.full((N_STATES, N_STATES), -1.0)

for i, j in edges:
    R[i, j] = 100 if j == GOAL else 0
    R[j, i] = 100 if i == GOAL else 0

R[GOAL, GOAL] = 100

# ---------------- Q-LEARNING PARAMETERS ----------------
Q = np.zeros((N_STATES, N_STATES))

gamma = 0.9        # 🔁 increased
alpha = 0.5        # 🔁 learning rate
epsilon = 0.6      # 🔁 exploration

# ---------------- FUNCTIONS ----------------
def available_actions(state):
    return np.where(R[state] >= 0)[0]

def choose_action(state):
    actions = available_actions(state)
    if random.random() < epsilon:
        return random.choice(actions)
    return actions[np.argmax(Q[state, actions])]

def update_q(state, action):
    best_next = np.max(Q[action])
    Q[state, action] = (1 - alpha) * Q[state, action] + \
                       alpha * (R[state, action] + gamma * best_next)

# ---------------- TRAINING ----------------
rewards = []

for episode in range(1500):   # 🔁 increased iterations
    state = random.randint(0, N_STATES-1)
    action = choose_action(state)
    update_q(state, action)

    rewards.append(np.mean(Q))
    epsilon = max(0.05, epsilon * 0.995)   # 🔁 decay exploration

print("Training completed.")

# ---------------- TESTING ----------------
current = 0
path = [current]

while current != GOAL:
    next_state = np.argmax(Q[current])
    path.append(next_state)
    current = next_state

print("Optimal Path to Goal:")
print(path)

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(8,4))
plt.plot(rewards)
plt.title("Q-Learning Convergence (Average Q-value)")
plt.xlabel("Episodes")
plt.ylabel("Average Q")
plt.grid(True)
plt.show()
