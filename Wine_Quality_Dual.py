import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
X = df.drop('quality', axis=1)
y = df['quality']

# Optional: Binary classification if needed
# y = (y >= 6).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== LAYER 1 ==========

# Base learners
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

# Train base learners
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Get predictions (label-level) from base learners
pred_knn_train = knn.predict(X_train).reshape(-1, 1)
pred_nb_train = nb.predict(X_train).reshape(-1, 1)

# Stack predictions to form input for meta learner (Decision Tree)
layer1_input_train = np.hstack([pred_knn_train, pred_nb_train])

# Meta learner of Layer 1
dt = DecisionTreeClassifier(random_state=42)
dt.fit(layer1_input_train, y_train)

# Predict on training set using Layer 1's DT (for Layer 2 input)
dt_train_output = dt.predict(layer1_input_train).reshape(-1, 1)

# ========== LAYER 2 ==========

# Meta learner: MLP
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=42)
mlp.fit(dt_train_output, y_train)

# ========== TESTING ==========

# Predict on test set using base learners
pred_knn_test = knn.predict(X_test).reshape(-1, 1)
pred_nb_test = nb.predict(X_test).reshape(-1, 1)
layer1_input_test = np.hstack([pred_knn_test, pred_nb_test])

# Predict using Layer 1 DT
dt_test_output = dt.predict(layer1_input_test).reshape(-1, 1)

# Final prediction from MLP
final_pred = mlp.predict(dt_test_output)

print(classification_report(y_test, final_pred, zero_division=1))