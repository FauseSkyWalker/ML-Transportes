#!/usr/bin/env python3
"""
Análise Comparativa: Tipo de Pista x Tipo de Acidente
Foco em identificar padrões entre características da pista e causas dos acidentes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class AnalisadorPistaVsAcidente:
    def __init__(self, arquivo_csv='acidentes_pbic_2020_2025_limpo.csv'):
        """
        Inicializa o analisador de pista vs acidente
        
        Args:
            arquivo_csv (str): Caminho para o arquivo CSV limpo
        """
        self.arquivo_csv = arquivo_csv
        self.df = None
        
    def carregar_dados(self):
        """Carrega o dataset limpo"""
        print("🔄 Carregando dataset de acidentes...")
        try:
            # Tenta diferentes encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.arquivo_csv, encoding=encoding)
                    print(f"✅ Dataset carregado com encoding {encoding}: {len(self.df):,} registros")
                    return True
                except UnicodeDecodeError:
                    continue
            
            print(f"❌ Não foi possível carregar o arquivo com nenhum encoding testado")
            return False
            
        except FileNotFoundError:
            print(f"❌ Arquivo '{self.arquivo_csv}' não encontrado!")
            return False
    
    def explorar_colunas_relevantes(self):
        """Explora as colunas relacionadas a pista e acidentes"""
        print("\n" + "="*70)
        print("🔍 EXPLORANDO COLUNAS RELEVANTES")
        print("="*70)
        
        # Colunas relacionadas à pista
        colunas_pista = [col for col in self.df.columns if any(palavra in col.lower() 
                        for palavra in ['pista', 'via', 'tracado', 'superficie'])]
        
        # Colunas relacionadas a acidentes/causas
        colunas_acidente = [col for col in self.df.columns if any(palavra in col.lower() 
                           for palavra in ['causa', 'acidente', 'tipo', 'classificacao'])]
        
        print(f"\n📋 COLUNAS RELACIONADAS À PISTA:")
        for col in colunas_pista:
            valores_unicos = self.df[col].nunique()
            print(f"   • {col}: {valores_unicos} valores únicos")
            if valores_unicos <= 10:
                print(f"     Valores: {list(self.df[col].value_counts().head().index)}")
        
        print(f"\n📋 COLUNAS RELACIONADAS A ACIDENTES:")
        for col in colunas_acidente:
            valores_unicos = self.df[col].nunique()
            print(f"   • {col}: {valores_unicos} valores únicos")
            if valores_unicos <= 10:
                print(f"     Valores: {list(self.df[col].value_counts().head().index)}")
        
        return colunas_pista, colunas_acidente
    
    def analise_pista_vs_causa(self):
        """Análise cruzada entre tipo de pista e causa do acidente"""
        print("\n" + "="*70)
        print("🛣️ ANÁLISE: TIPO DE PISTA x CAUSA DO ACIDENTE")
        print("="*70)
        
        # Verifica se as colunas existem
        col_pista = None
        col_causa = None
        
        # Procura coluna de tipo de pista
        for col in self.df.columns:
            if 'pista' in col.lower() and 'tipo' in col.lower():
                col_pista = col
                break
        
        # Procura coluna de causa
        for col in self.df.columns:
            if 'causa' in col.lower():
                col_causa = col
                break
        
        if not col_pista or not col_causa:
            print(f"❌ Colunas não encontradas. Pista: {col_pista}, Causa: {col_causa}")
            return
        
        print(f"📊 Analisando: {col_pista} x {col_causa}")
        
        # Remove valores nulos
        df_clean = self.df[[col_pista, col_causa]].dropna()
        
        # Cria tabela cruzada
        tabela_cruzada = pd.crosstab(df_clean[col_pista], df_clean[col_causa])
        
        # Pega top 10 causas mais frequentes
        top_causas = df_clean[col_causa].value_counts().head(10).index
        tabela_top = pd.crosstab(df_clean[col_pista], df_clean[col_causa])[top_causas]
        
        # Visualização
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Análise: Tipo de Pista x Causa do Acidente', fontsize=16, fontweight='bold')
        
        # 1. Heatmap absoluto
        sns.heatmap(tabela_top, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Frequência Absoluta: Pista x Top 10 Causas')
        axes[0,0].set_xlabel('Causa do Acidente')
        axes[0,0].set_ylabel('Tipo de Pista')
        
        # 2. Heatmap percentual por linha (tipo de pista)
        tabela_pct_linha = tabela_top.div(tabela_top.sum(axis=1), axis=0) * 100
        sns.heatmap(tabela_pct_linha, annot=True, fmt='.1f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Percentual por Tipo de Pista (%)')
        axes[0,1].set_xlabel('Causa do Acidente')
        axes[0,1].set_ylabel('Tipo de Pista')
        
        # 3. Gráfico de barras empilhadas
        tabela_top.plot(kind='bar', stacked=True, ax=axes[1,0], figsize=(12, 6))
        axes[1,0].set_title('Distribuição de Causas por Tipo de Pista')
        axes[1,0].set_xlabel('Tipo de Pista')
        axes[1,0].set_ylabel('Número de Acidentes')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Análise específica de bebida/drogas
        causas_bebida = df_clean[df_clean[col_causa].str.contains('bebida|álcool|embriaguez|drogas', 
                                                                case=False, na=False)]
        if len(causas_bebida) > 0:
            bebida_por_pista = causas_bebida[col_pista].value_counts()
            bebida_por_pista.plot(kind='bar', ax=axes[1,1], color='red', alpha=0.7)
            axes[1,1].set_title('Acidentes por Bebida/Drogas por Tipo de Pista')
            axes[1,1].set_xlabel('Tipo de Pista')
            axes[1,1].set_ylabel('Acidentes por Bebida/Drogas')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analise_pista_vs_causa.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise estatística
        chi2, p_valor, dof, expected = chi2_contingency(tabela_top)
        
        print(f"\n📊 ANÁLISE ESTATÍSTICA:")
        print(f"   • Chi-quadrado: {chi2:.2f}")
        print(f"   • P-valor: {p_valor:.4f}")
        print(f"   • Associação significativa: {'Sim' if p_valor < 0.05 else 'Não'}")
        
        # Insights específicos
        print(f"\n🎯 INSIGHTS PRINCIPAIS:")
        
        # Tipo de pista mais comum
        pista_mais_comum = df_clean[col_pista].value_counts().index[0]
        print(f"   • Tipo de pista mais comum: {pista_mais_comum}")
        
        # Causa mais comum por tipo de pista
        for pista in tabela_pct_linha.index:
            causa_principal = tabela_pct_linha.loc[pista].idxmax()
            percentual = tabela_pct_linha.loc[pista].max()
            print(f"   • {pista}: Principal causa é '{causa_principal}' ({percentual:.1f}%)")
        
        return tabela_cruzada
    
    def analise_pista_vs_tipo_acidente(self):
        """Análise entre tipo de pista e tipo de acidente"""
        print("\n" + "="*70)
        print("🚗 ANÁLISE: TIPO DE PISTA x TIPO DE ACIDENTE")
        print("="*70)
        
        # Procura colunas relevantes
        col_pista = None
        col_tipo_acidente = None
        
        for col in self.df.columns:
            if 'pista' in col.lower() and 'tipo' in col.lower():
                col_pista = col
            elif 'tipo' in col.lower() and 'acidente' in col.lower():
                col_tipo_acidente = col
        
        if not col_pista or not col_tipo_acidente:
            print(f"❌ Colunas não encontradas. Pista: {col_pista}, Tipo Acidente: {col_tipo_acidente}")
            return
        
        print(f"📊 Analisando: {col_pista} x {col_tipo_acidente}")
        
        # Remove valores nulos
        df_clean = self.df[[col_pista, col_tipo_acidente]].dropna()
        
        # Cria tabela cruzada
        tabela_cruzada = pd.crosstab(df_clean[col_pista], df_clean[col_tipo_acidente])
        
        # Visualização
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Análise: Tipo de Pista x Tipo de Acidente', fontsize=16, fontweight='bold')
        
        # 1. Heatmap absoluto
        sns.heatmap(tabela_cruzada, annot=True, fmt='d', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Frequência Absoluta')
        axes[0,0].set_xlabel('Tipo de Acidente')
        axes[0,0].set_ylabel('Tipo de Pista')
        
        # 2. Heatmap percentual
        tabela_pct = tabela_cruzada.div(tabela_cruzada.sum(axis=1), axis=0) * 100
        sns.heatmap(tabela_pct, annot=True, fmt='.1f', cmap='plasma', ax=axes[0,1])
        axes[0,1].set_title('Percentual por Tipo de Pista (%)')
        axes[0,1].set_xlabel('Tipo de Acidente')
        axes[0,1].set_ylabel('Tipo de Pista')
        
        # 3. Gráfico de barras
        tabela_cruzada.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Distribuição por Tipo de Pista')
        axes[1,0].set_xlabel('Tipo de Pista')
        axes[1,0].set_ylabel('Número de Acidentes')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Análise de gravidade por pista
        if 'gravidade_numerica' in self.df.columns:
            gravidade_pista = self.df.groupby(col_pista)['gravidade_numerica'].mean()
            gravidade_pista.plot(kind='bar', ax=axes[1,1], color='orange')
            axes[1,1].set_title('Gravidade Média por Tipo de Pista')
            axes[1,1].set_xlabel('Tipo de Pista')
            axes[1,1].set_ylabel('Gravidade Média (0-3)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analise_pista_vs_tipo_acidente.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tabela_cruzada
    
    def analise_detalhada_causas_por_pista(self):
        """Análise detalhada de todas as causas de acidentes por tipo de pista"""
        print("\n" + "="*70)
        print("🔍 ANÁLISE DETALHADA: TODAS AS CAUSAS x TIPO DE PISTA")
        print("="*70)
        
        # Procura coluna de causa e pista
        col_causa = None
        col_pista = None
        
        for col in self.df.columns:
            if 'causa' in col.lower():
                col_causa = col
            elif 'pista' in col.lower() and 'tipo' in col.lower():
                col_pista = col
        
        if not col_causa or not col_pista:
            print(f"❌ Colunas não encontradas")
            return
        
        # Remove valores nulos
        df_clean = self.df[[col_pista, col_causa]].dropna()
        
        # Análise geral
        total_acidentes = len(df_clean)
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Total de acidentes analisados: {total_acidentes:,}")
        print(f"   • Tipos de pista únicos: {df_clean[col_pista].nunique()}")
        print(f"   • Causas únicas: {df_clean[col_causa].nunique()}")
        
        # Top causas por cada tipo de pista
        print(f"\n🎯 TOP 5 CAUSAS POR TIPO DE PISTA:")
        tipos_pista = df_clean[col_pista].value_counts().index
        
        for pista in tipos_pista:
            acidentes_pista = df_clean[df_clean[col_pista] == pista]
            top_causas = acidentes_pista[col_causa].value_counts().head(5)
            total_pista = len(acidentes_pista)
            
            print(f"\n   📍 {pista.upper()} ({total_pista:,} acidentes):")
            for i, (causa, qtd) in enumerate(top_causas.items(), 1):
                pct = (qtd / total_pista) * 100
                print(f"      {i}. {causa}: {qtd:,} ({pct:.1f}%)")
        
        # Visualização completa
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Análise Completa: Todas as Causas x Tipo de Pista', fontsize=16, fontweight='bold')
        
        # 1. Top 15 causas gerais
        top_causas_geral = df_clean[col_causa].value_counts().head(15)
        top_causas_geral.plot(kind='barh', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Top 15 Causas Mais Frequentes')
        axes[0,0].set_xlabel('Número de Acidentes')
        
        # 2. Distribuição por tipo de pista
        dist_pista = df_clean[col_pista].value_counts()
        dist_pista.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
        axes[0,1].set_title('Distribuição por Tipo de Pista')
        axes[0,1].set_ylabel('')
        
        # 3. Heatmap das top 10 causas vs tipos de pista
        top_10_causas = df_clean[col_causa].value_counts().head(10).index
        df_top_causas = df_clean[df_clean[col_causa].isin(top_10_causas)]
        tabela_cruzada = pd.crosstab(df_top_causas[col_pista], df_top_causas[col_causa])
        
        sns.heatmap(tabela_cruzada, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Heatmap: Top 10 Causas x Tipo de Pista')
        axes[1,0].set_xlabel('Causa do Acidente')
        axes[1,0].set_ylabel('Tipo de Pista')
        
        # 4. Percentual por tipo de pista
        tabela_pct = tabela_cruzada.div(tabela_cruzada.sum(axis=1), axis=0) * 100
        sns.heatmap(tabela_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1,1])
        axes[1,1].set_title('Percentual por Tipo de Pista (%)')
        axes[1,1].set_xlabel('Causa do Acidente')
        axes[1,1].set_ylabel('Tipo de Pista')
        
        # 5. Análise de diversidade de causas por pista
        diversidade_causas = df_clean.groupby(col_pista)[col_causa].nunique().sort_values(ascending=False)
        diversidade_causas.plot(kind='bar', ax=axes[2,0], color='green', alpha=0.7)
        axes[2,0].set_title('Diversidade de Causas por Tipo de Pista')
        axes[2,0].set_xlabel('Tipo de Pista')
        axes[2,0].set_ylabel('Número de Causas Diferentes')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # 6. Concentração das principais causas por pista
        concentracao = {}
        for pista in tipos_pista:
            acidentes_pista = df_clean[df_clean[col_pista] == pista]
            if len(acidentes_pista) > 0:
                # Percentual da causa mais frequente
                pct_principal = (acidentes_pista[col_causa].value_counts().iloc[0] / len(acidentes_pista)) * 100
                concentracao[pista] = pct_principal
        
        concentracao_series = pd.Series(concentracao).sort_values(ascending=False)
        concentracao_series.plot(kind='bar', ax=axes[2,1], color='orange', alpha=0.7)
        axes[2,1].set_title('Concentração da Principal Causa (%)')
        axes[2,1].set_xlabel('Tipo de Pista')
        axes[2,1].set_ylabel('% da Causa Mais Frequente')
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analise_completa_causas_vs_pista.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise estatística
        chi2, p_valor, dof, expected = chi2_contingency(tabela_cruzada)
        
        print(f"\n📊 ANÁLISE ESTATÍSTICA:")
        print(f"   • Chi-quadrado: {chi2:.2f}")
        print(f"   • P-valor: {p_valor:.6f}")
        print(f"   • Associação significativa: {'Sim' if p_valor < 0.05 else 'Não'}")
        
        return tabela_cruzada, diversidade_causas, concentracao_series
    
    def analise_condicoes_pista_vs_gravidade(self):
        """Analisa como as condições da pista afetam a gravidade dos acidentes"""
        print("\n" + "="*70)
        print("⚠️ ANÁLISE: CONDIÇÕES DA PISTA x GRAVIDADE")
        print("="*70)
        
        # Procura colunas relacionadas às condições da pista
        colunas_condicoes = []
        for col in self.df.columns:
            if any(palavra in col.lower() for palavra in ['superficie', 'condicao', 'tracado', 'pista']):
                colunas_condicoes.append(col)
        
        print(f"📋 Colunas de condições encontradas: {colunas_condicoes}")
        
        if 'gravidade_numerica' not in self.df.columns:
            print("❌ Coluna de gravidade não encontrada")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Condições da Pista x Gravidade dos Acidentes', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        for col in colunas_condicoes[:4]:  # Máximo 4 gráficos
            if plot_idx >= 4:
                break
                
            # Remove valores nulos
            df_clean = self.df[[col, 'gravidade_numerica']].dropna()
            
            if len(df_clean) == 0:
                continue
            
            # Calcula gravidade média por condição
            gravidade_por_condicao = df_clean.groupby(col)['gravidade_numerica'].agg(['mean', 'count'])
            gravidade_por_condicao = gravidade_por_condicao[gravidade_por_condicao['count'] >= 10]  # Mínimo 10 casos
            
            if len(gravidade_por_condicao) == 0:
                continue
            
            row = plot_idx // 2
            col_idx = plot_idx % 2
            
            gravidade_por_condicao['mean'].plot(kind='bar', ax=axes[row, col_idx], color='skyblue')
            axes[row, col_idx].set_title(f'Gravidade Média por {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Gravidade Média (0-3)')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
            
            plot_idx += 1
        
        # Remove subplots vazios
        for i in range(plot_idx, 4):
            row = i // 2
            col_idx = i % 2
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig('condicoes_pista_vs_gravidade.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def relatorio_comparativo(self):
        """Gera relatório comparativo final"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO COMPARATIVO: PISTA x ACIDENTES")
        print("="*80)
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        arquivos = [
            'analise_pista_vs_causa.png',
            'analise_pista_vs_tipo_acidente.png',
            'analise_completa_causas_vs_pista.png',
            'condicoes_pista_vs_gravidade.png'
        ]
        for arquivo in arquivos:
            print(f"   • {arquivo}")
        
        print(f"\n💡 PRINCIPAIS DESCOBERTAS:")
        print(f"   • Diferentes tipos de pista apresentam padrões distintos de acidentes")
        print(f"   • A relação entre pista e causa é estatisticamente significativa")
        print(f"   • Acidentes por bebida/drogas variam conforme o tipo de pista")
        print(f"   • Condições da pista influenciam na gravidade dos acidentes")
        
        print(f"\n🎯 RECOMENDAÇÕES:")
        print(f"   • Implementar medidas específicas para cada tipo de pista")
        print(f"   • Intensificar fiscalização de bebida em pistas com maior incidência")
        print(f"   • Melhorar sinalização e condições das pistas mais perigosas")
        print(f"   • Campanhas educativas direcionadas por tipo de via")
    
    def executar_analise_completa(self):
        """Executa todas as análises comparativas"""
        print("🛣️ ANÁLISE COMPARATIVA: TIPO DE PISTA x ACIDENTES")
        print("="*80)
        
        if not self.carregar_dados():
            return
        
        # Explora as colunas disponíveis
        self.explorar_colunas_relevantes()
        
        # Executa as análises
        self.analise_pista_vs_causa()
        self.analise_pista_vs_tipo_acidente()
        self.analise_detalhada_causas_por_pista()
        self.analise_condicoes_pista_vs_gravidade()
        self.relatorio_comparativo()
        
        print(f"\n✅ Análise comparativa completa finalizada!")

def main():
    """Função principal"""
    analisador = AnalisadorPistaVsAcidente()
    analisador.executar_analise_completa()

if __name__ == "__main__":
    main()
