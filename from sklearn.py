from sklearn.linear_model import LogisticRegression

# Criar e treinar o modelo de Regressão Logística
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Fazer predições
y_pred_logreg = logreg_model.predict(X_test_scaled)

# Avaliar o modelo
print("Relatório de Classificação - Regressão Logística:")
print(classification_report(y_test, y_pred_logreg))