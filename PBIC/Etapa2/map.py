import pandas as pd
import folium

# Carregar CSV
df = pd.read_csv('acidentes_prf_filtrado.csv')

# Criar mapa centralizado no Brasil
m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)

# Adicionar marcadores
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"UF: {row['uf']}<br>"
               f"BR: {row['br']}<br>"
               f"Município: {row['municipio']}<br>"
               f"Tipo: {row['tipo_acidente']}<br>"
               f"Classificação: {row['classificacao_acidente']}<br>"
               f"Pista: {row['tipo_pista']}")
    ).add_to(m)

# Salvar mapa
m.save('mapa_acidentes.html')
print("Mapa gerado: abra o arquivo 'mapa_acidentes.html' no navegador")
