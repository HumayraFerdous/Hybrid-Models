import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE,VarianceThreshold
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
data =pd.read_csv("train.csv")
#print(data.isnull().sum())
data['binary_label']=(data['label']==0).astype(int)
print(data.head())
X = data.drop(columns=['label','binary_label'])
y_binary = data['binary_label']


selector = VarianceThreshold(threshold=0.0)
X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
kbest = SelectKBest(score_func=f_classif, k=30)
X_kbest = kbest.fit_transform(X, y_binary)
selected_kbest_features = X.columns[kbest.get_support()]

rfe = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=15)
X_rfe = rfe.fit_transform(X[selected_kbest_features], y_binary)
final_features = selected_kbest_features[rfe.get_support()]

print(f"Selected Features: {list(final_features)}")

X_train, X_test, y_train, y_test = train_test_split(X[final_features], y_binary, test_size=0.2, random_state=42)

lgb = LGBMClassifier()
lgb.fit(X_train, y_train)
y_pred_binary = lgb.predict(X_test)

print("=== First Layer: Binary Classification ===")
print(classification_report(y_test, y_pred_binary, target_names=['Other Digits', 'Digit 0']))

data['pred_binary'] = lgb.predict(X[final_features])
df_layer2 = data[(data['pred_binary'] == 0) & (data['label']!=0)] # Only samples predicted as not '0'

X_layer2 = df_layer2[final_features]
y_layer2 = df_layer2['label']  # multiclass labels: 1–9
label_map = {old: new for new, old in enumerate(sorted(y_layer2.unique()))}
reverse_map = {v: k for k, v in label_map.items()}
y_layer2_mapped = y_layer2.map(label_map)
# Train/test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X_layer2, y_layer2_mapped, test_size=0.2, random_state=42)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Stacking ensemble
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# Train and predict
stacked_model.fit(X2_train, y2_train)
y_pred_stack = stacked_model.predict(X2_test)
y_pred_original = [reverse_map[pred] for pred in y_pred_stack]
y_true_original = [reverse_map[true] for true in y2_test]
# Evaluation
from sklearn.metrics import classification_report
print("=== Second Layer: Multiclass Classification with Stacked Ensemble ===")
print(classification_report(y_true_original, y_pred_original))