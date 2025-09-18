#!/usr/bin/env python3
"""
Análise do Dataset Sem Nulos PESID - PBIC
Script para analisar o dataset removendo registros com pesid nulo
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configurações
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

def analisar_dataset_sem_nulos():
    """Analisa o dataset removendo registros com pesid nulo"""
    print("🔍 ANÁLISE DO DATASET SEM NULOS EM PESID")
    print("="*60)
    
    # Carrega o dataset limpo
    try:
        df = pd.read_csv('acidentes_pbic_2020_2025_limpo.csv')
        print(f"✅ Dataset original carregado: {len(df):,} registros")
        print(f"📊 Total de colunas: {len(df.columns)}")
        
        # Remove registros com pesid nulo
        df_filtrado = df.dropna(subset=['pesid'])
        registros_removidos = len(df) - len(df_filtrado)
        percentual_removido = (registros_removidos / len(df)) * 100
        
        print(f"📉 Registros removidos: {registros_removidos:,} ({percentual_removido:.1f}%)")
        print(f"✅ Dataset filtrado: {len(df_filtrado):,} registros")
        
    except FileNotFoundError:
        print("❌ Arquivo 'acidentes_pbic_2020_2025_limpo.csv' não encontrado!")
        return
    
    # Verifica valores nulos
    print(f"\n🔍 VALORES NULOS POR COLUNA:")
    valores_nulos = df_filtrado.isnull().sum()
    colunas_com_nulos = valores_nulos[valores_nulos > 0].sort_values(ascending=False)
    
    if len(colunas_com_nulos) > 0:
        for coluna, qtd_nulos in colunas_com_nulos.items():
            percentual = (qtd_nulos / len(df_filtrado)) * 100
            print(f"   • {coluna}: {qtd_nulos:,} ({percentual:.1f}%)")
    else:
        print("   ✅ Nenhuma coluna com valores nulos!")
    
    # Acidentes únicos vs pessoas
    print(f"\n🚗 ACIDENTES vs PESSOAS:")
    if 'id' in df_filtrado.columns:
        acidentes_unicos = df_filtrado['id'].nunique()
        pessoas_total = len(df_filtrado)
        media_pessoas_por_acidente = pessoas_total / acidentes_unicos
        
        print(f"   • Acidentes únicos: {acidentes_unicos:,}")
        print(f"   • Pessoas envolvidas: {pessoas_total:,}")
        print(f"   • Média pessoas/acidente: {media_pessoas_por_acidente:.1f}")
    
    # Distribuição por ano
    print(f"\n📅 DISTRIBUIÇÃO POR ANO:")
    if 'ano_arquivo' in df_filtrado.columns:
        dist_ano = df_filtrado['ano_arquivo'].value_counts().sort_index()
        for ano, qtd in dist_ano.items():
            print(f"   • {ano}: {qtd:,} pessoas")
    
    # Distribuição de gravidade
    if 'gravidade_numerica' in df_filtrado.columns:
        print(f"\n🚑 DISTRIBUIÇÃO DE GRAVIDADE:")
        gravidade_labels = {0: 'Sem Vítimas', 1: 'Feridos Leves', 2: 'Feridos Graves', 3: 'Mortos'}
        dist_gravidade = df_filtrado['gravidade_numerica'].value_counts().sort_index()
        
        for nivel, qtd in dist_gravidade.items():
            label = gravidade_labels.get(nivel, f'Nível {nivel}')
            percentual = (qtd / len(df_filtrado)) * 100
            print(f"   • {label}: {qtd:,} ({percentual:.1f}%)")
    
    print(f"\n✅ Análise concluída!")

if __name__ == "__main__":
    analisar_dataset_sem_nulos()
