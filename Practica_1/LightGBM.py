# -*- coding: utf-8 -*-
"""
Este código procesa el dataset y entrena dos modelos de LightGBM de forma separada:
  - Uno para predecir la resistencia a Erythromycin.
  - Otro para predecir la resistencia a Ciprofloxacin.
Se entrena con early stopping y se plotean las curvas de pérdida (log loss) y las curvas ROC.
"""

# LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import lightgbm as lgb

# Lectura de datos y creación de variables
df = pd.read_csv("./practica_micro.csv")

df['Clase'] = np.where((df['Erythromycin'] == 1) & (df['Ciprofloxacin'] == 1), 3.0,
                       np.where(df['Erythromycin'] == 1, 1,
                                np.where(df['Ciprofloxacin'] == 1, 2, 0)))

# Expansión de la columna "MALDI_binned"

df['MALDI_binned'] = df['MALDI_binned'].apply(eval)
maldi_data = df['MALDI_binned'].apply(lambda x: pd.Series(x))
maldi_data = maldi_data.add_prefix('Proteina_')
df = pd.concat([df, maldi_data], axis=1)
df.drop(columns=['MALDI_binned'], inplace=True)


# Preprocesado de las intensidades

protein_cols = [col for col in df.columns if col.startswith('Proteina_')]

# Aplicar umbral y escalar
umbral = 0.00005
factor = 100
df[protein_cols] = df[protein_cols].map(lambda x: 0 if x < umbral else x)
df[protein_cols] = df[protein_cols] * factor

scaler = StandardScaler()
df[protein_cols] = scaler.fit_transform(df[protein_cols])


# División del conjunto

data, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Clase'])
train, val = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Clase'])

x_train = train[protein_cols]
x_val   = val[protein_cols]
x_test  = test[protein_cols]

y_train_erythro = train['Erythromycin']
y_val_erythro   = val['Erythromycin']
y_test_erythro  = test['Erythromycin']

y_train_cipro = train['Ciprofloxacin']
y_val_cipro   = val['Ciprofloxacin']
y_test_cipro  = test['Ciprofloxacin']


# MODELO CON LIGHTGBM PARA ERYTHROMYCIN

train_set_erythro = lgb.Dataset(x_train, label=y_train_erythro)
val_set_erythro = lgb.Dataset(x_val, label=y_val_erythro, reference=train_set_erythro)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'seed': 42,
    'verbose': -1
}

evals_result_erythro = {}

bst_erythro = lgb.train(params,
                        train_set_erythro,
                        num_boost_round=1000,
                        valid_sets=[train_set_erythro, val_set_erythro],
                        callbacks=[lgb.record_evaluation(evals_result_erythro),
                                   lgb.early_stopping(20)])

print(f"Mejor iteración para Erythromycin: {bst_erythro.best_iteration}")

# Ploteo de la evolución de la pérdida para Erythromycin
train_loss = evals_result_erythro['training']['binary_logloss']
val_loss = evals_result_erythro['valid_1']['binary_logloss']
plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Iteraciones')
plt.ylabel('Binary Logloss')
plt.title('Curva de Pérdida - Erythromycin')
plt.legend()
plt.show()

# Evaluación en test para Erythromycin
y_test_pred_proba_erythro = bst_erythro.predict(x_test, num_iteration=bst_erythro.best_iteration)
y_test_pred_erythro = (y_test_pred_proba_erythro >= 0.5).astype(int)
print("Test Accuracy (Erythromycin):", accuracy_score(y_test_erythro, y_test_pred_erythro))
print("Test AUROC (Erythromycin):", roc_auc_score(y_test_erythro, y_test_pred_proba_erythro))


# MODELO CON LIGHTGBM PARA CIPROFLOXACIN

train_set_cipro = lgb.Dataset(x_train, label=y_train_cipro)
val_set_cipro = lgb.Dataset(x_val, label=y_val_cipro, reference=train_set_cipro)

evals_result_cipro = {}

bst_cipro = lgb.train(params,
                      train_set_cipro,
                      num_boost_round=1000,
                      valid_sets=[train_set_cipro, val_set_cipro],
                      callbacks=[lgb.record_evaluation(evals_result_cipro),
                                 lgb.early_stopping(20)])

print(f"Mejor iteración para Ciprofloxacin: {bst_cipro.best_iteration}")

# Ploteo de la evolución de la pérdida para Ciprofloxacin
train_loss_cipro = evals_result_cipro['training']['binary_logloss']
val_loss_cipro = evals_result_cipro['valid_1']['binary_logloss']
plt.figure(figsize=(8,5))
plt.plot(train_loss_cipro, label='Train Loss')
plt.plot(val_loss_cipro, label='Validation Loss')
plt.xlabel('Iteraciones')
plt.ylabel('Binary Logloss')
plt.title('Curva de Pérdida - Ciprofloxacin')
plt.legend()
plt.show()

# Evaluación en test para Ciprofloxacin
y_test_pred_proba_cipro = bst_cipro.predict(x_test, num_iteration=bst_cipro.best_iteration)
y_test_pred_cipro = (y_test_pred_proba_cipro >= 0.5).astype(int)
print("Test Accuracy (Ciprofloxacin):", accuracy_score(y_test_cipro, y_test_pred_cipro))
print("Test AUROC (Ciprofloxacin):", roc_auc_score(y_test_cipro, y_test_pred_proba_cipro))
