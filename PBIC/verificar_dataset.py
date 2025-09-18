#!/usr/bin/env python3
"""
Verificação do Dataset Limpo - PBIC
Script para analisar o dataset gerado
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configurações
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

def analisar_dataset():
    """Analisa o dataset limpo gerado"""
    print("🔍 ANÁLISE DO DATASET LIMPO")
    print("="*50)
    
    # Carrega o dataset limpo
    try:
        df = pd.read_csv('acidentes_pbic_2020_2025_limpo.csv')
        print(f"✅ Dataset carregado: {len(df):,} registros")
        print(f"📊 Total de colunas: {len(df.columns)}")
    except FileNotFoundError:
        print("❌ Arquivo 'acidentes_pbic_2020_2025_limpo.csv' não encontrado!")
        return
    
    # Mostra as colunas
    print(f"\n📋 COLUNAS DO DATASET:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Verifica valores nulos
    print(f"\n🔍 VALORES NULOS POR COLUNA:")
    valores_nulos = df.isnull().sum()
    colunas_com_nulos = valores_nulos[valores_nulos > 0].sort_values(ascending=False)
    
    if len(colunas_com_nulos) > 0:
        for coluna, qtd_nulos in colunas_com_nulos.items():
            percentual = (qtd_nulos / len(df)) * 100
            print(f"   • {coluna}: {qtd_nulos:,} ({percentual:.1f}%)")
    else:
        print("   ✅ Nenhuma coluna com valores nulos!")
    
    # Acidentes únicos vs pessoas
    print(f"\n🚗 ACIDENTES vs PESSOAS:")
    if 'id' in df.columns:
        acidentes_unicos = df['id'].nunique()
        pessoas_total = len(df)
        media_pessoas_por_acidente = pessoas_total / acidentes_unicos
        
        print(f"   • Acidentes únicos: {acidentes_unicos:,}")
        print(f"   • Pessoas envolvidas: {pessoas_total:,}")
        print(f"   • Média pessoas/acidente: {media_pessoas_por_acidente:.1f}")
    
    # Distribuição por ano
    print(f"\n📅 DISTRIBUIÇÃO POR ANO:")
    if 'ano_arquivo' in df.columns:
        dist_ano = df['ano_arquivo'].value_counts().sort_index()
        for ano, qtd in dist_ano.items():
            print(f"   • {ano}: {qtd:,} pessoas")
    
    # Estatísticas das novas colunas criadas
    print(f"\n🆕 ESTATÍSTICAS DAS NOVAS COLUNAS:")
    
    novas_colunas = ['ano_arquivo', 'mes', 'dia_mes', 'hora', 'gravidade_numerica', 'total_vitimas']
    for col in novas_colunas:
        if col in df.columns:
            print(f"\n   📊 {col}:")
            if df[col].dtype in ['int64', 'float64']:
                print(f"      Min: {df[col].min()}")
                print(f"      Max: {df[col].max()}")
                print(f"      Média: {df[col].mean():.2f}")
            else:
                print(f"      Valores únicos: {df[col].nunique()}")
    
    # Distribuição de gravidade
    if 'gravidade_numerica' in df.columns:
        print(f"\n🚑 DISTRIBUIÇÃO DE GRAVIDADE:")
        gravidade_labels = {0: 'Sem Vítimas', 1: 'Feridos Leves', 2: 'Feridos Graves', 3: 'Mortos'}
        dist_gravidade = df['gravidade_numerica'].value_counts().sort_index()
        
        for nivel, qtd in dist_gravidade.items():
            label = gravidade_labels.get(nivel, f'Nível {nivel}')
            percentual = (qtd / len(df)) * 100
            print(f"   • {label}: {qtd:,} ({percentual:.1f}%)")
    
    print(f"\n✅ Análise concluída!")

if __name__ == "__main__":
    analisar_dataset()
