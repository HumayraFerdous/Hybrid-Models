from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== First Layer: MLP ======
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Use MLP probabilities as input to second layer
mlp_train_probs = mlp.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
mlp_test_probs = mlp.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

# ====== Second Layer: KNN ======
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(mlp_train_probs, y_train)

# Final prediction
final_pred = knn.predict(mlp_test_probs)

# ====== Evaluation ======
accuracy = accuracy_score(y_test, final_pred)
print("Stacked Ensemble Accuracy (MLP → KNN):", accuracy)