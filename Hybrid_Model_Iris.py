from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Base classifiers
knn = KNeighborsClassifier()
gnb = GaussianNB()

# Define hybrid model (weights to be tuned)
hybrid = VotingClassifier(estimators=[
    ('knn', knn),
    ('gnb', gnb)
], voting='hard')

# Define grid of weights to try
param_grid = {
    'weights': [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
}

# Grid search
grid = GridSearchCV(estimator=hybrid, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

# Best result
print(f"Best weights: {grid.best_params_['weights']}")
print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")