from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base learners
base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('gnb', GaussianNB())
]

# Define meta-learner
meta_learner = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Stacking classifier
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False  # Set to True to give original features to meta-learner too
)

# Train and evaluate
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)
accuracy = classification_report(y_test, y_pred)
print("Stacked Ensemble Accuracy: ",accuracy)
