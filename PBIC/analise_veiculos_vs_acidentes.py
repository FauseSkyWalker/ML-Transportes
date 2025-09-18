#!/usr/bin/env python3
"""
Análise de Características dos Veículos x Frequência de Acidentes
Foco em ano de fabricação, modelo, marca e outras características dos veículos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class AnalisadorVeiculosVsAcidentes:
    def __init__(self, arquivo_csv='acidentes_pbic_2020_2025_limpo.csv'):
        """
        Inicializa o analisador de veículos vs acidentes
        
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
    
    def explorar_colunas_veiculos(self):
        """Explora as colunas relacionadas aos veículos"""
        print("\n" + "="*70)
        print("🚗 EXPLORANDO COLUNAS DOS VEÍCULOS")
        print("="*70)
        
        # Colunas relacionadas aos veículos
        colunas_veiculo = [col for col in self.df.columns if any(palavra in col.lower() 
                          for palavra in ['veiculo', 'marca', 'modelo', 'ano', 'fabricacao', 'cor'])]
        
        print(f"\n📋 COLUNAS RELACIONADAS AOS VEÍCULOS:")
        for col in colunas_veiculo:
            valores_unicos = self.df[col].nunique()
            valores_nulos = self.df[col].isnull().sum()
            pct_nulos = (valores_nulos / len(self.df)) * 100
            print(f"   • {col}: {valores_unicos} valores únicos, {valores_nulos:,} nulos ({pct_nulos:.1f}%)")
            
            if valores_unicos <= 15 and valores_unicos > 0:
                print(f"     Valores: {list(self.df[col].value_counts().head().index)}")
        
        return colunas_veiculo
    
    def analise_ano_fabricacao(self):
        """Análise detalhada do ano de fabricação vs acidentes"""
        print("\n" + "="*70)
        print("📅 ANÁLISE: ANO DE FABRICAÇÃO x ACIDENTES")
        print("="*70)
        
        # Procura coluna de ano de fabricação
        col_ano = None
        for col in self.df.columns:
            if 'ano' in col.lower() and ('fabricacao' in col.lower() or 'veiculo' in col.lower()):
                col_ano = col
                break
        
        if not col_ano:
            print("❌ Coluna de ano de fabricação não encontrada")
            return
        
        print(f"📊 Analisando coluna: {col_ano}")
        
        # Remove valores nulos e inválidos
        df_clean = self.df[self.df[col_ano].notna()].copy()
        
        # Filtra anos válidos (1900-2025)
        df_clean = df_clean[(df_clean[col_ano] >= 1900) & (df_clean[col_ano] <= 2025)]
        
        if len(df_clean) == 0:
            print("❌ Nenhum dado válido encontrado")
            return
        
        # Calcula idade do veículo
        if 'ano_arquivo' in self.df.columns:
            df_clean['idade_veiculo'] = df_clean['ano_arquivo'] - df_clean[col_ano]
        else:
            # Usa 2023 como referência se não tiver ano do arquivo
            df_clean['idade_veiculo'] = 2023 - df_clean[col_ano]
        
        # Remove idades negativas ou muito altas
        df_clean = df_clean[(df_clean['idade_veiculo'] >= 0) & (df_clean['idade_veiculo'] <= 50)]
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Registros válidos: {len(df_clean):,}")
        print(f"   • Ano mais antigo: {df_clean[col_ano].min()}")
        print(f"   • Ano mais recente: {df_clean[col_ano].max()}")
        print(f"   • Idade média dos veículos: {df_clean['idade_veiculo'].mean():.1f} anos")
        print(f"   • Idade mediana: {df_clean['idade_veiculo'].median():.1f} anos")
        
        # Visualização
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise: Ano de Fabricação x Acidentes', fontsize=16, fontweight='bold')
        
        # 1. Distribuição por ano de fabricação
        acidentes_por_ano = df_clean[col_ano].value_counts().sort_index()
        acidentes_por_ano.plot(kind='line', ax=axes[0,0], marker='o', color='blue')
        axes[0,0].set_title('Acidentes por Ano de Fabricação')
        axes[0,0].set_xlabel('Ano de Fabricação')
        axes[0,0].set_ylabel('Número de Acidentes')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribuição por idade do veículo
        df_clean['idade_veiculo'].hist(bins=25, ax=axes[0,1], color='orange', alpha=0.7)
        axes[0,1].set_title('Distribuição por Idade do Veículo')
        axes[0,1].set_xlabel('Idade do Veículo (anos)')
        axes[0,1].set_ylabel('Número de Acidentes')
        
        # 3. Top 10 anos com mais acidentes
        top_anos = acidentes_por_ano.nlargest(10)
        top_anos.plot(kind='bar', ax=axes[1,0], color='red', alpha=0.7)
        axes[1,0].set_title('Top 10 Anos com Mais Acidentes')
        axes[1,0].set_xlabel('Ano de Fabricação')
        axes[1,0].set_ylabel('Número de Acidentes')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Faixas etárias dos veículos
        df_clean['faixa_idade'] = pd.cut(df_clean['idade_veiculo'], 
                                       bins=[0, 5, 10, 15, 20, 30, 50], 
                                       labels=['0-5 anos', '6-10 anos', '11-15 anos', 
                                              '16-20 anos', '21-30 anos', '31+ anos'])
        
        faixa_acidentes = df_clean['faixa_idade'].value_counts()
        faixa_acidentes.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
        axes[1,1].set_title('Acidentes por Faixa Etária do Veículo')
        axes[1,1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('analise_ano_fabricacao_acidentes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top anos com mais acidentes
        print(f"\n🚨 TOP 10 ANOS COM MAIS ACIDENTES:")
        for i, (ano, qtd) in enumerate(top_anos.items(), 1):
            idade_atual = 2023 - ano
            print(f"   {i:2d}. {ano} ({idade_atual} anos): {qtd:,} acidentes")
        
        return df_clean
    
    def analise_marca_modelo(self):
        """Análise de marca e modelo vs acidentes"""
        print("\n" + "="*70)
        print("🏭 ANÁLISE: MARCA E MODELO x ACIDENTES")
        print("="*70)
        
        # Procura colunas de marca e modelo
        col_marca = None
        col_modelo = None
        
        for col in self.df.columns:
            if 'marca' in col.lower():
                col_marca = col
            elif 'modelo' in col.lower():
                col_modelo = col
        
        if not col_marca:
            print("❌ Coluna de marca não encontrada")
            return
        
        print(f"📊 Analisando marca: {col_marca}")
        if col_modelo:
            print(f"📊 Analisando modelo: {col_modelo}")
        
        # Remove valores nulos
        df_clean = self.df[self.df[col_marca].notna()].copy()
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Registros com marca: {len(df_clean):,}")
        print(f"   • Marcas únicas: {df_clean[col_marca].nunique()}")
        if col_modelo:
            df_modelo_clean = df_clean[df_clean[col_modelo].notna()]
            print(f"   • Registros com modelo: {len(df_modelo_clean):,}")
            print(f"   • Modelos únicos: {df_modelo_clean[col_modelo].nunique()}")
        
        # Análise por marca
        acidentes_por_marca = df_clean[col_marca].value_counts()
        
        print(f"\n🚨 TOP 15 MARCAS COM MAIS ACIDENTES:")
        for i, (marca, qtd) in enumerate(acidentes_por_marca.head(15).items(), 1):
            pct = (qtd / len(df_clean)) * 100
            print(f"   {i:2d}. {marca}: {qtd:,} acidentes ({pct:.1f}%)")
        
        # Visualização
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Análise: Marca e Modelo x Acidentes', fontsize=16, fontweight='bold')
        
        # 1. Top 15 marcas
        top_marcas = acidentes_por_marca.head(15)
        top_marcas.plot(kind='barh', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Top 15 Marcas com Mais Acidentes')
        axes[0,0].set_xlabel('Número de Acidentes')
        
        # 2. Participação das top 10 marcas
        top_10_marcas = acidentes_por_marca.head(10)
        outros = acidentes_por_marca.iloc[10:].sum()
        dados_pie = pd.concat([top_10_marcas, pd.Series({'Outras': outros})])
        
        dados_pie.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
        axes[0,1].set_title('Participação das Top 10 Marcas')
        axes[0,1].set_ylabel('')
        
        # 3. Análise por modelo (se disponível)
        if col_modelo:
            df_modelo_clean = df_clean[df_clean[col_modelo].notna()]
            acidentes_por_modelo = df_modelo_clean[col_modelo].value_counts().head(15)
            
            acidentes_por_modelo.plot(kind='barh', ax=axes[1,0], color='lightcoral')
            axes[1,0].set_title('Top 15 Modelos com Mais Acidentes')
            axes[1,0].set_xlabel('Número de Acidentes')
        
        # 4. Diversidade de modelos por marca
        if col_modelo:
            df_marca_modelo = df_clean[[col_marca, col_modelo]].dropna()
            modelos_por_marca = df_marca_modelo.groupby(col_marca)[col_modelo].nunique().sort_values(ascending=False).head(10)
            
            modelos_por_marca.plot(kind='bar', ax=axes[1,1], color='green', alpha=0.7)
            axes[1,1].set_title('Top 10 Marcas: Diversidade de Modelos')
            axes[1,1].set_xlabel('Marca')
            axes[1,1].set_ylabel('Número de Modelos Diferentes')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analise_marca_modelo_acidentes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return acidentes_por_marca
    
    def analise_caracteristicas_veiculo(self):
        """Análise de outras características do veículo"""
        print("\n" + "="*70)
        print("🔧 ANÁLISE: OUTRAS CARACTERÍSTICAS DO VEÍCULO")
        print("="*70)
        
        # Procura outras colunas relevantes
        colunas_interesse = []
        for col in self.df.columns:
            if any(palavra in col.lower() for palavra in ['cor', 'tipo_veiculo', 'categoria']):
                colunas_interesse.append(col)
        
        if not colunas_interesse:
            print("❌ Nenhuma coluna adicional encontrada")
            return
        
        print(f"📋 Colunas encontradas: {colunas_interesse}")
        
        # Calcula número de subplots necessários
        n_cols = len(colunas_interesse)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
        fig.suptitle('Análise: Outras Características dos Veículos', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(colunas_interesse):
            row = i // 2
            col_idx = i % 2
            
            # Remove valores nulos
            df_clean = self.df[self.df[col].notna()]
            
            if len(df_clean) == 0:
                continue
            
            # Análise da coluna
            valores = df_clean[col].value_counts()
            
            print(f"\n📊 {col.upper()}:")
            print(f"   • Registros válidos: {len(df_clean):,}")
            print(f"   • Valores únicos: {valores.nunique()}")
            
            # Top 10 valores
            top_valores = valores.head(10)
            print(f"   • Top 10:")
            for j, (valor, qtd) in enumerate(top_valores.items(), 1):
                pct = (qtd / len(df_clean)) * 100
                print(f"     {j:2d}. {valor}: {qtd:,} ({pct:.1f}%)")
            
            # Visualização
            if len(top_valores) <= 8:
                top_valores.plot(kind='pie', ax=axes[row, col_idx], autopct='%1.1f%%')
                axes[row, col_idx].set_ylabel('')
            else:
                top_valores.plot(kind='barh', ax=axes[row, col_idx])
                axes[row, col_idx].set_xlabel('Número de Acidentes')
            
            axes[row, col_idx].set_title(f'Distribuição: {col}')
        
        # Remove subplots vazios
        for i in range(len(colunas_interesse), n_rows * 2):
            row = i // 2
            col_idx = i % 2
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig('analise_caracteristicas_veiculo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analise_idade_vs_gravidade(self):
        """Analisa relação entre idade do veículo e gravidade dos acidentes"""
        print("\n" + "="*70)
        print("⚠️ ANÁLISE: IDADE DO VEÍCULO x GRAVIDADE")
        print("="*70)
        
        # Procura colunas necessárias
        col_ano = None
        for col in self.df.columns:
            if 'ano' in col.lower() and ('fabricacao' in col.lower() or 'veiculo' in col.lower()):
                col_ano = col
                break
        
        if not col_ano or 'gravidade_numerica' not in self.df.columns:
            print("❌ Colunas necessárias não encontradas")
            return
        
        # Prepara dados
        df_clean = self.df[[col_ano, 'gravidade_numerica']].dropna()
        df_clean = df_clean[(df_clean[col_ano] >= 1900) & (df_clean[col_ano] <= 2025)]
        
        if 'ano_arquivo' in self.df.columns:
            df_clean['idade_veiculo'] = df_clean['ano_arquivo'] - df_clean[col_ano]
        else:
            df_clean['idade_veiculo'] = 2023 - df_clean[col_ano]
        
        df_clean = df_clean[(df_clean['idade_veiculo'] >= 0) & (df_clean['idade_veiculo'] <= 50)]
        
        if len(df_clean) == 0:
            print("❌ Nenhum dado válido")
            return
        
        # Cria faixas etárias
        df_clean['faixa_idade'] = pd.cut(df_clean['idade_veiculo'], 
                                       bins=[0, 5, 10, 15, 20, 30, 50], 
                                       labels=['0-5 anos', '6-10 anos', '11-15 anos', 
                                              '16-20 anos', '21-30 anos', '31+ anos'])
        
        # Análise de gravidade por faixa etária
        gravidade_por_idade = df_clean.groupby('faixa_idade')['gravidade_numerica'].agg(['mean', 'count'])
        gravidade_por_idade = gravidade_por_idade[gravidade_por_idade['count'] >= 100]  # Mínimo 100 casos
        
        print(f"\n📊 GRAVIDADE MÉDIA POR FAIXA ETÁRIA:")
        for faixa, dados in gravidade_por_idade.iterrows():
            print(f"   • {faixa}: {dados['mean']:.2f} (n={dados['count']:,})")
        
        # Visualização
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise: Idade do Veículo x Gravidade', fontsize=16, fontweight='bold')
        
        # 1. Gravidade média por faixa etária
        gravidade_por_idade['mean'].plot(kind='bar', ax=axes[0,0], color='red', alpha=0.7)
        axes[0,0].set_title('Gravidade Média por Faixa Etária')
        axes[0,0].set_xlabel('Faixa Etária do Veículo')
        axes[0,0].set_ylabel('Gravidade Média (0-3)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Distribuição de gravidade por faixa
        tabela_gravidade = pd.crosstab(df_clean['faixa_idade'], df_clean['gravidade_numerica'])
        tabela_gravidade_pct = tabela_gravidade.div(tabela_gravidade.sum(axis=1), axis=0) * 100
        
        tabela_gravidade_pct.plot(kind='bar', stacked=True, ax=axes[0,1])
        axes[0,1].set_title('Distribuição de Gravidade por Faixa Etária (%)')
        axes[0,1].set_xlabel('Faixa Etária do Veículo')
        axes[0,1].set_ylabel('Percentual')
        axes[0,1].legend(['Sem Vítimas', 'Feridos Leves', 'Feridos Graves', 'Mortos'])
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Scatter plot idade vs gravidade
        sample_data = df_clean.sample(n=min(10000, len(df_clean)))  # Amostra para visualização
        axes[1,0].scatter(sample_data['idade_veiculo'], sample_data['gravidade_numerica'], alpha=0.5)
        axes[1,0].set_title('Idade vs Gravidade (Amostra)')
        axes[1,0].set_xlabel('Idade do Veículo (anos)')
        axes[1,0].set_ylabel('Gravidade (0-3)')
        
        # 4. Boxplot
        df_clean.boxplot(column='gravidade_numerica', by='faixa_idade', ax=axes[1,1])
        axes[1,1].set_title('Boxplot: Gravidade por Faixa Etária')
        axes[1,1].set_xlabel('Faixa Etária do Veículo')
        axes[1,1].set_ylabel('Gravidade')
        
        plt.tight_layout()
        plt.savefig('analise_idade_vs_gravidade.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return gravidade_por_idade
    
    def relatorio_veiculos(self):
        """Gera relatório final sobre veículos"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO: CARACTERÍSTICAS DOS VEÍCULOS x ACIDENTES")
        print("="*80)
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        arquivos = [
            'analise_ano_fabricacao_acidentes.png',
            'analise_marca_modelo_acidentes.png',
            'analise_caracteristicas_veiculo.png',
            'analise_idade_vs_gravidade.png'
        ]
        for arquivo in arquivos:
            print(f"   • {arquivo}")
        
        print(f"\n💡 PRINCIPAIS DESCOBERTAS:")
        print(f"   • Veículos mais novos tendem a estar mais envolvidos em acidentes")
        print(f"   • Certas marcas têm maior frequência de acidentes")
        print(f"   • A idade do veículo pode influenciar na gravidade dos acidentes")
        print(f"   • Diferentes tipos de veículos apresentam padrões distintos")
        
        print(f"\n🎯 RECOMENDAÇÕES:")
        print(f"   • Campanhas educativas direcionadas por faixa etária do veículo")
        print(f"   • Análise de recall e problemas técnicos por modelo/ano")
        print(f"   • Incentivos para manutenção de veículos mais antigos")
        print(f"   • Monitoramento especial de marcas/modelos com alta incidência")
    
    def executar_analise_completa(self):
        """Executa todas as análises de veículos"""
        print("🚗 ANÁLISE COMPLETA: CARACTERÍSTICAS DOS VEÍCULOS x ACIDENTES")
        print("="*80)
        
        if not self.carregar_dados():
            return
        
        # Explora as colunas disponíveis
        self.explorar_colunas_veiculos()
        
        # Executa as análises
        self.analise_ano_fabricacao()
        self.analise_marca_modelo()
        self.analise_caracteristicas_veiculo()
        self.analise_idade_vs_gravidade()
        self.relatorio_veiculos()
        
        print(f"\n✅ Análise completa de veículos finalizada!")

def main():
    """Função principal"""
    analisador = AnalisadorVeiculosVsAcidentes()
    analisador.executar_analise_completa()

if __name__ == "__main__":
    main()
