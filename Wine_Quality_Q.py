import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('winequality-red.csv')  # or white wine
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Discretize state space (simplify)
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X).astype(int)

# Combine all features into a tuple to represent state
states = [tuple(x) for x in X_discrete]

# Define actions (quality ratings range)
actions = list(range(y.min(), y.max() + 1))

# Initialize Q-table: (state, action) -> Q-value
from collections import defaultdict
Q = defaultdict(lambda: np.zeros(len(actions)))

# Parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2
episodes = 10000

# Train
for _ in range(episodes):
    idx = np.random.randint(0, len(states))
    state = states[idx]
    true_quality = y[idx]

    if np.random.rand() < epsilon:
        action = np.random.choice(actions)
    else:
        action = actions[np.argmax(Q[state])]

    reward = 1 if action == true_quality else -1
    next_state = state  # no transition here

    Q[state][actions.index(action)] += alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state][actions.index(action)]
    )

# Evaluation
correct = 0
for i in range(len(states)):
    state = states[i]
    predicted = actions[np.argmax(Q[state])]
    if predicted == y[i]:
        correct += 1

accuracy = correct / len(states)
print(f"Toy Q-learning Accuracy: {accuracy:.2f}")