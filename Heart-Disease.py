import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,roc_auc_score


data = pd.read_csv("heart.csv")
#print(data.head())
#print(data.isnull().sum())
X = data.drop('target',axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=chi2,k=8)
X_kbest = selector.fit_transform(X,y)
selected_features_filter = selector.get_support(indices=True)
print("Filter-selected features: ",X.columns[selected_features_filter])

estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=8)
X_rfe = rfe.fit_transform(X_scaled, y)
selected_features_wrapper = rfe.get_support(indices=True)
print("Wrapper-selected features:", X.columns[selected_features_wrapper])
# Combine selected features (intersection or union)
selected_indices = list(set(selected_features_filter).union(set(selected_features_wrapper)))
X_selected = X_scaled[:, selected_indices]
print("Final selected features:", X.columns[selected_indices])

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, kernel='rbf')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, probs):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()