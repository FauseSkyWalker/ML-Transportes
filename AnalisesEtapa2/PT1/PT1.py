#%% 
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


sns.set_theme()
#%%
df = pd.read_excel('/home/fause/ML-Transportes/Dados/analiseestatistica_indicadores2.xlsx')
#%%
# Pré-processamento
df['km_rodovias_federais'] = df['km_rodovias_federais'].fillna(0)
df['area_km2'] = df['area_km2'].fillna(df['area_km2'].median())
#df = pd.get_dummies(df, columns=['UF', 'Estado'], drop_first=True)
df['Integrado ao SNT'] = df['Integrado ao SNT'].map({'Sim': 1, 'Não': 0})
df['log_sinistros'] = np.log1p(df['Sinistros'])  # log(1 + x)
#%%
features = [
    'PIB per capita', 'Tx de Alfabetização +15 anos',
    'Taxa de Admissão em Empregos', 'IDHM',
    'Taxa de Óbitos/100 mil habitantes', '% de Óbitos/Sinistros',
    'area_km2', 'km_rodovias_federais', 'Integrado ao SNT'
] 

X = df[features]
y = df['log_sinistros']
#%%
# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputação e normalização
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
#%%
print("Sinistros (min, max, média, mediana):")
print(df['Sinistros'].min(), df['Sinistros'].max(), df['Sinistros'].mean(), df['Sinistros'].median())
#%%
# Regressão Linear

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
#%%
# b Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

#%%
# c) XGBoost Regressor

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
#%%
import pandas as pd

# Supondo que você tenha um DataFrame original X com colunas:
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Criar explainer usando o DataFrame
explainer = shap.Explainer(xgb, X_test_df)

# Calcular valores SHAP
shap_values = explainer(X_test_df)

# Agora plotar sem passar argumento 'features'
plt.title("SHAP Summary Plot — impacto das variáveis em log(Sinistros)")
shap.plots.beeswarm(shap_values)
plt.show()

# %%
import shap

# Explainer e valores SHAP (você já tem)
explainer = shap.Explainer(xgb, X_test_df)
shap_values = explainer(X_test_df)

# Importância média (bar plot)
shap.summary_plot(shap_values, X_test_df, plot_type='bar')

# Dependence plot para a feature mais importante (exemplo 'PIB per capita')
shap.dependence_plot('PIB per capita', shap_values.values, X_test_df)

# %%
shap_interaction_values = shap.TreeExplainer(xgb).shap_interaction_values(X_test_df)
shap.summary_plot(shap_interaction_values, X_test_df)
# %%
plt.figure(figsize=(10,8))
sns.heatmap(df[features + ['log_sinistros']].corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação")
plt.show()

# %%
