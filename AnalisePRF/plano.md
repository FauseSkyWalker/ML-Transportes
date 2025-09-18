
---

## 3. Metodologia

### 3.1 Coleta e Consolidação
1. Baixar os dados anuais (2020 a 2025) no formato **por pessoa com causa**.
2. Padronizar nomes de colunas e tipos de dados.
3. Concatenar em uma base única para análise.
4. Salvar em formato `.csv` limpo para uso posterior.

---

### 3.2 Avaliação de Qualidade dos Dados
- **Completude**: medir percentual de valores ausentes por coluna.
- **Consistência**: verificar se formatos de data, categorias e valores numéricos estão corretos.
- **Padronização geográfica**: uniformizar nomes de municípios e UFs.
- **Detecção de outliers**: identificar valores extremos (ex.: idade negativa, km > limite da BR).

---

### 3.3 Análise Exploratória (EDA)
- Distribuição de acidentes por:
- Ano, mês, dia da semana e horário.
- Estado e município.
- Tipo e causa de acidente.
- Condição meteorológica, tipo de pista e uso do solo.
- Mapas de calor georreferenciados (latitude e longitude).
- Correlações entre variáveis (ex.: tipo de veículo x gravidade).

---

### 3.4 Modelagem de Machine Learning
#### Objetivo
Prever gravidade do acidente (por exemplo: **fatal** vs **não fatal**).

#### Abordagem sugerida
1. **Modelos clássicos**:
 - Random Forest
 - Gradient Boosting (XGBoost, LightGBM, CatBoost)
2. **Comparação com Deep Learning (MLP)**:
 - Apenas como teste, para avaliar ganho real.
 - Usar normalização e regularização para evitar overfitting.

#### Variáveis-alvo e preditoras
- **Alvo**: `mortos` (ou classificação derivada, ex.: fatal/não fatal).
- **Preditoras**: todas as variáveis categóricas e numéricas relevantes (excluindo IDs).

---

### 3.5 Validação
- Divisão treino/teste (70/30) ou validação cruzada.
- Métricas:
- Acurácia
- Precisão / Recall
- F1-score
- Matriz de confusão
- Avaliação de importância de variáveis (feature importance).

---

## 4. Entregáveis
- Base de dados única (2020–2025) limpa e padronizada.
- Relatório de **qualidade dos dados**.
- **EDA** com gráficos e insights.
- Comparativo de desempenho entre modelos clássicos e deep learning.
- Recomendações para aplicação em escala nacional.

---

## 5. Observações
- Deep learning será avaliado, mas modelos clássicos devem ser priorizados inicialmente.
- Futuras integrações podem incluir:
- Dados meteorológicos históricos.
- Informações de fluxo de tráfego.
- Textos livres dos boletins (NLP).


Para Classificação:

Prever gravidade do acidente (ilesos/feridos/mortos)
Prever tipo de acidente baseado em condições
Algoritmos: Random Forest, XGBoost, SVM
Para Regressão:

Prever número de vítimas
Algoritmos: Random Forest, Gradient Boosting
Para Análise Exploratória:

Clustering para identificar padrões de acidentes
Análise temporal para sazonalidades