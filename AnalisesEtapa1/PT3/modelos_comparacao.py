# Comparação de Diferentes Modelos com Análise de Importância das Features

# 1. Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance, features, title):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    print(f"\nImportância das Features para {title} (%):")
    print(feature_importance.assign(importance=lambda x: x['importance']*100).round(2))

# Regressão Logística
print("=== Regressão Logística ===")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled)
lr_pred = lr_model.predict(X_test_scaled)
print("\nRelatório de Classificação:")
print(classification_report(y_test, lr_pred))

# Importância das features para Regressão Logística usando coeficientes absolutos
lr_importance = np.abs(lr_model.coef_[0])
plot_feature_importance(lr_importance, features, 'Regressão Logística')

# Naive Bayes
print("\n=== Naive Bayes ===")
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)
nb_pred = nb_model.predict(X_test_scaled)
print("\nRelatório de Classificação:")
print(classification_report(y_test, nb_pred))

# Importância das features para Naive Bayes usando permutation importance
nb_importance = permutation_importance(nb_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
plot_feature_importance(nb_importance.importances_mean, features, 'Naive Bayes')

# XGBoost
print("\n=== XGBoost ===")
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_pred = xgb_model.predict(X_test_scaled)
print("\nRelatório de Classificação:")
print(classification_report(y_test, xgb_pred))

# Importância das features para XGBoost
plot_feature_importance(xgb_model.feature_importances_, features, 'XGBoost')

# Comparação das Importâncias entre Modelos
importance_comparison = pd.DataFrame({
    'Feature': features,
    'Logistic_Regression': lr_importance,
    'Naive_Bayes': nb_importance.importances_mean,
    'XGBoost': xgb_model.feature_importances_,
    'Random_Forest': rf_model.feature_importances_  # usando o rf_model já treinado anteriormente
})

# Normalizar as importâncias para cada modelo
for col in ['Logistic_Regression', 'Naive_Bayes', 'XGBoost', 'Random_Forest']:
    importance_comparison[col] = importance_comparison[col] / importance_comparison[col].sum()

# Criar heatmap de comparação
plt.figure(figsize=(12, 8))
sns.heatmap(importance_comparison.set_index('Feature').T, 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd')
plt.title('Comparação da Importância das Features entre Modelos')
plt.tight_layout()
plt.show()
