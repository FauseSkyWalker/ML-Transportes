#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()  # ou sns.set()
#----------------------------------------
#%%
# Carregar os dados
df = pd.read_excel('/home/fause/ML-Transportes/Dados/analiseestatistica_indicadores2.xlsx') 
print("Dimensões do dataset:", df.shape)
df.head()
df.tail()

#----------------------------------------
#%%
# Informações gerais do dataframe
print(df.info())

# Estatísticas descritivas das colunas numéricas
print(df.describe())

# Verificando valores nulos por coluna
print(df.isnull().sum())
#----------------------------------------
#%%
#tratamento de dados

#*km_rodovias_federais* Essa variável pode naturalmente ser 0 para muitos municípios que não têm rodovias federais passando por eles.
df['km_rodovias_federais'] = df['km_rodovias_federais'].fillna(0)

#*km_rodovias_estaduais* PReencher com a mediana das áreas — ela é menos sensível a outliers que a média.
df['area_km2'] = df['area_km2'].fillna(df['area_km2'].median())

# Boxplots para detectar outliers(Nenhum Outlier agravante)
plt.figure(figsize=(15, 5))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.show()

#variaveis categoricas 

# Codificando a variável categórica 'Integrado ao SNT' transformando em 0(não) ou 1(Sim)
# encoder = LabelEncoder()
# df['Integrado ao SNT'] = encoder.fit_transform(df['Integrado ao SNT'])

#codificando a variaveis UF e Estado

#Essa abordagem cria uma nova coluna para cada categoria,
# onde o valor é 1 se a categoria está presente e 0 caso contrário.
#  É útil quando não há uma relação ordenada entre as categorias.
df = pd.get_dummies(df, columns=['UF', 'Estado'], drop_first=True)

#----------------------------------------
#%%
print("Linhas:", len(df))
print("Colunas:", len(df.columns))


# %%

# Definir features (X) e target (y)
features = ['Sinistros','PIB per capita', 'Tx de Alfabetização +15 anos', 
           'Taxa de Admissão em Empregos', 'IDHM', 
           'Taxa de Óbitos/100 mil habitantes', '% de Óbitos/Sinistros','area_km2', 'km_rodovias_federais'
           ]

#'area_km2', 'km_rodovias_federais'

X = df[features]
y = df['Integrado ao SNT']
y = y.map({'Sim': 1, 'Não': 0}) 

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


imputer = SimpleImputer(strategy='median') # Imputação de valores faltantes
scaler = StandardScaler() # Normalização dos dados, padronizando a média e o desvio padrão

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# %%

# Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Treinando o modelo #n_estimators=200
rf_model = RandomForestClassifier(n_estimators=500, class_weight=None, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predição e avaliação
y_pred = rf_model.predict(X_test_scaled)

print("\nRelatório de Classificação após SMOTE:\n")
print(classification_report(y_test, y_pred))

# %%
importances = rf_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices], y=feature_names[sorted_indices])
plt.title("Importância das Features - Random Forest")
plt.show()

#%%
# Calcular e plotar importância das features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importância das Features para Integração ao SNT')
plt.tight_layout()
plt.show()

print("\nImportância das Features (%):\n")
print(feature_importance.assign(importance=lambda x: x['importance']*100).round(2))
#%%
# %%
# Matriz de correlação

# 🧠 O que é uma matriz de correlação?
# É uma tabela que mostra a força e direção da relação entre variáveis numéricas. Cada valor da matriz representa um coeficiente de correlação de Pearson, que varia de:

# +1 → correlação positiva perfeita (quando uma variável aumenta, a outra também)

# 0 → sem correlação

# –1 → correlação negativa perfeita (quando uma variável aumenta, a outra diminui)
plt.figure(figsize=(12, 10))
correlation_matrix = df[features + ['Integrado ao SNT']].copy()
correlation_matrix['Integrado ao SNT'] = y  # garantindo que a coluna esteja binarizada

corr = correlation_matrix.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Matriz de confusão

# Geração da matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Rótulos personalizados para as células
labels = [
    ["Verdadeiro Negativo", "Falso Positivo"],
    ["Falso Negativo", "Verdadeiro Positivo"]
]

# Criar nova matriz com labels nas células
annot = [[f"{label}\n{cm[i][j]}" for j, label in enumerate(row)] for i, row in enumerate(labels)]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, linewidths=1, linecolor='black')
plt.title('Matriz de Confusão com Rótulos Explicativos')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.xticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.yticks(ticks=[0.5, 1.5], labels=['Negativo', 'Positivo'])
plt.show()
#%%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
#%%
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
#%%
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
#%%

# %%
# Filtrar municípios com população maior que 20 mil (Medio + Grande + metropole)
df_acima_100k = df[df['População'] > 20000] #20000:


#%%
# Features e target
X = df_acima_100k[features]
y = df_acima_100k['Integrado ao SNT']

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputação e normalização
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
# Aplicar SMOTE para balancear as classes após a filtragem
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#
# %%
# Modelo
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predição e avaliação
y_pred = rf_model.predict(X_test_scaled)
print("\nRelatório de Classificação — População > 20 mil:")
print(classification_report(y_test, y_pred))
# %%

# Importância das features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importância das Features — População > 100 mil')
plt.tight_layout()
plt.show()

# %%

