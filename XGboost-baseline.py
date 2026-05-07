import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             roc_auc_score, precision_score, recall_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# ============================================================
# ÉTAPE 1 : Chargement + Concaténation
# ============================================================
print("Chargement des datasets...")

apt   = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\aptmini (1).csv")
oil   = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\oilrig_labeled_final4.csv")
data1 = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\training_data.csv");  data1['label'] = 0
data2 = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\training_data2.csv"); data2['label'] = 0
data3 = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\training_data3.csv"); data3['label'] = 0
data4 = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\training_data4.csv"); data4['label'] = 0
data5 = pd.read_csv(r"C:\Users\dell\MyDataset\MyEDR\training_data5.csv"); data5['label'] = 0

df = pd.concat([apt, oil, data1, data2, data3, data4, data5], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset total : {df.shape}")
print(f"Distribution labels :\n{df['label'].value_counts()}")

# ============================================================
# ÉTAPE 2 : Nettoyage
# ============================================================
cols_to_drop = [
    'timestamp', 'type', 'pid', 'ppid', 'process_guid', 'image_name', 'command_line',
    'source', 'is_office_app', 'has_encoded_command', 'registry_rename_count_log',
    'time_anomaly_score', 'file_entropy_change', 'day_sin', 'day_cos', 'session_norm'
]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.drop_duplicates(inplace=True)
print(f"Shape après nettoyage : {df.shape}")

# ============================================================
# ÉTAPE 3 : Split 80% / 15% / 5%
# ============================================================
X = df.drop(columns=cols_to_drop + ['label'], errors='ignore')
X = X.select_dtypes(include=[np.number])
y = df['label']

RATIO = y.value_counts()[0] / y.value_counts()[1]
print(f"\nRatio déséquilibre : {RATIO:.2f}:1")
print(f"(Baseline : on ignore ce déséquilibre volontairement)")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\nTrain      : {X_train.shape} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation : {X_val.shape} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test       : {X_test.shape} ({len(X_test)/len(X)*100:.0f}%)")

# ============================================================
# ÉTAPE 4 : Entraînement XGBoost BASELINE (sans scale_pos_weight)
# ============================================================
print("\n>>> XGBoost BASELINE (sans scale_pos_weight)")

xgb_baseline = xgb.XGBClassifier(
    n_estimators=200,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    early_stopping_rounds=20,
    verbosity=0
    # Pas de scale_pos_weight → modèle ignore le déséquilibre
)

t0 = time.time()
xgb_baseline.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)
train_time = time.time() - t0

# ============================================================
# ÉTAPE 5 : Courbes Training Loss vs Validation Loss
# ============================================================
results    = xgb_baseline.evals_result()
train_loss = results['validation_0']['logloss']
val_loss   = results['validation_1']['logloss']
best_round = xgb_baseline.best_iteration

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss',   color='steelblue', linewidth=2)
plt.plot(range(1, len(val_loss)+1),   val_loss,   label='Validation Loss', color='tomato',    linewidth=2)
plt.axvline(x=best_round, color='green', linestyle='--', label=f'Best round ({best_round})')
plt.title('Baseline — Training Loss vs Validation Loss', fontsize=14)
plt.xlabel('Rounds')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_baseline.png', dpi=150)
plt.show()
print("Courbe sauvegardée : loss_baseline.png")

# ============================================================
# ÉTAPE 6 : Évaluation sur Test
# ============================================================
t1 = time.time()
y_pred  = xgb_baseline.predict(X_test)
y_proba = xgb_baseline.predict_proba(X_test)[:, 1]
infer_time_batch = time.time() - t1

# Temps réel sur 1 événement (moyenne sur 1000 appels)
single_event = X_test.iloc[[0]]
t_single = time.time()
for _ in range(1000):
    xgb_baseline.predict(single_event)
time_single = (time.time() - t_single) / 1000

print(f"\n{'='*60}")
print(f"  RÉSULTATS BASELINE — TEST SET (5%)")
print(f"{'='*60}")
print(f"\nTemps entraînement      : {train_time:.2f}s")
print(f"Nombre de rounds        : {best_round}")
print(f"Temps/round             : {train_time/best_round*1000:.2f} ms")
print(f"Temps/événement (batch) : {infer_time_batch/len(y_test)*1000:.4f} ms")
print(f"Temps/événement (réel)  : {time_single*1000:.4f} ms")

print(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1        : {f1_score(y_test, y_pred, average='binary', zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nTP : {tp}  — attaques bien détectées")
print(f"TN : {tn}  — bénins bien classés")
print(f"FP : {fp}  — bénins classés comme attaque")
print(f"FN : {fn}  — attaques manquées  ← important pour un EDR")

# Matrice de confusion
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Bénin', 'Malveillant'],
            yticklabels=['Bénin', 'Malveillant'])
plt.title('Matrice de Confusion — XGBoost Baseline')
plt.ylabel('Vrai label')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('confusion_baseline.png', dpi=150)
plt.show()
print("Matrice sauvegardée : confusion_baseline.png")

# Top 15 features
feat_imp = pd.Series(xgb_baseline.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(15)
plt.figure(figsize=(9, 6))
feat_imp.plot(kind='barh', color='steelblue')
plt.title('Top 15 Features — XGBoost Baseline')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('features_baseline.png', dpi=150)
plt.show()
print("Features sauvegardées : features_baseline.png")

# ============================================================
# EXPORT ONNX
# ============================================================
feature_names = list(X.columns)
with open("features_baseline.json", "w") as f:
    json.dump(feature_names, f, indent=2)
print(f"\nfeatures_baseline.json sauvegardé ({len(feature_names)} features)")

X_train_renamed = X_train.copy()
X_train_renamed.columns = [f"f{i}" for i in range(X_train.shape[1])]

xgb_export = xgb.XGBClassifier(
    n_estimators=best_round,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_export.fit(X_train_renamed, y_train)

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_xgboost(xgb_export, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, "apt_detector_baseline.onnx")
print("apt_detector_baseline.onnx sauvegardé")
print("\nTerminé — Baseline !")