# ===========================================================
# train_institutional_vix75_ai.py
# ===========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from joblib import dump

print("Training Institutional Ensemble AI for VIX75 (5M Entry)")

# Load Data
df = pd.read_csv("data/vix75_merged_enriched.csv")
df.dropna(inplace=True)

# Create Target Labels
df['future_close'] = df['close'].shift(-3)
df['signal'] = np.where(df['future_close'] > df['close'] * 1.001, 1,
                        np.where(df['future_close'] < df['close'] * 0.999, -1, 0))
df.dropna(inplace=True)

# Label Mapping
label_map = {-1: 0, 0: 1, 1: 2}
df['signal'] = df['signal'].map(label_map)

# Feature Selection
exclude = ['timestamp', 'future_close', 'signal']
X = df.drop(columns=exclude)
y = df['signal']

# Split + Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base Models
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=500, learning_rate=0.05)
lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=40)

# Stacking Ensemble
estimators = [('rf', rf), ('xgb', xgb), ('lgbm', lgbm)]
stack = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100, max_depth=6))

print("Training...")
stack.fit(X_train, y_train)

# Evaluation
y_pred = stack.predict(X_test)
print("\n===== MODEL PERFORMANCE =====")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model + scaler
dump(stack, "institutional_vix75_model.joblib")
dump(scaler, "institutional_scaler.joblib")
print("Saved model and scaler for live prediction.")
print("Institutional Ensemble AI training complete.")