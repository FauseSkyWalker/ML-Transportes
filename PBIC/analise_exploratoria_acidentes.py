#!/usr/bin/env python3
"""
Análise Exploratória de Acidentes - PBIC (2020-2025)
Script para identificar padrões e análises de segurança viária
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AnalisadorPadroesAcidentes:
    def __init__(self, arquivo_csv='acidentes_pbic_2020_2025_limpo.csv'):
        """
        Inicializa o analisador de padrões de acidentes
        
        Args:
            arquivo_csv (str): Caminho para o arquivo CSV limpo
        """
        self.arquivo_csv = arquivo_csv
        self.df = None
        
    def carregar_dados(self):
        """Carrega o dataset limpo"""
        print("🔄 Carregando dataset de acidentes...")
        try:
            self.df = pd.read_csv(self.arquivo_csv)
            print(f"✅ Dataset carregado: {len(self.df):,} registros")
            return True
        except FileNotFoundError:
            print(f"❌ Arquivo '{self.arquivo_csv}' não encontrado!")
            return False
    
    def analise_temporal_detalhada(self):
        """Análise temporal: horários e dias de maior risco"""
        print("\n" + "="*70)
        print("⏰ ANÁLISE TEMPORAL - HORÁRIOS E DIAS DE MAIOR RISCO")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise Temporal dos Acidentes', fontsize=16, fontweight='bold')
        
        # 1. Acidentes por hora do dia
        if 'hora' in self.df.columns:
            acidentes_hora = self.df.groupby('hora').size()
            acidentes_hora.plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Acidentes por Hora do Dia')
            axes[0,0].set_xlabel('Hora')
            axes[0,0].set_ylabel('Número de Acidentes')
            axes[0,0].tick_params(axis='x', rotation=0)
            
            # Identifica horários mais perigosos
            top_horas = acidentes_hora.nlargest(3)
            print(f"\n🚨 HORÁRIOS MAIS PERIGOSOS:")
            for hora, qtd in top_horas.items():
                print(f"   • {hora:02d}:00h - {qtd:,} acidentes")
        
        # 2. Acidentes por dia da semana
        if 'dia_semana' in self.df.columns:
            acidentes_dia = self.df['dia_semana'].value_counts()
            ordem_dias = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 
                         'sexta-feira', 'sábado', 'domingo']
            acidentes_dia = acidentes_dia.reindex([d for d in ordem_dias if d in acidentes_dia.index])
            
            acidentes_dia.plot(kind='bar', ax=axes[0,1], color='lightcoral')
            axes[0,1].set_title('Acidentes por Dia da Semana')
            axes[0,1].set_xlabel('Dia da Semana')
            axes[0,1].set_ylabel('Número de Acidentes')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Heatmap hora vs dia da semana
        if 'hora' in self.df.columns and 'dia_semana' in self.df.columns:
            heatmap_data = self.df.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)
            if len(heatmap_data) > 0:
                sns.heatmap(heatmap_data, ax=axes[1,0], cmap='YlOrRd', cbar_kws={'label': 'Número de Acidentes'})
                axes[1,0].set_title('Heatmap: Hora vs Dia da Semana')
                axes[1,0].set_xlabel('Hora')
                axes[1,0].set_ylabel('Dia da Semana')
        
        # 4. Acidentes por mês (sazonalidade)
        if 'mes' in self.df.columns:
            acidentes_mes = self.df['mes'].value_counts().sort_index()
            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                    'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            acidentes_mes.index = [meses[i-1] for i in acidentes_mes.index]
            
            acidentes_mes.plot(kind='line', ax=axes[1,1], marker='o', color='green')
            axes[1,1].set_title('Sazonalidade dos Acidentes')
            axes[1,1].set_xlabel('Mês')
            axes[1,1].set_ylabel('Número de Acidentes')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analise_temporal_acidentes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: analise_temporal_acidentes.png")
    
    def analise_trechos_perigosos(self):
        """Identifica trechos mais perigosos por BR e km"""
        print("\n" + "="*70)
        print("🛣️ ANÁLISE DE TRECHOS PERIGOSOS")
        print("="*70)
        
        # Análise por BR
        if 'br' in self.df.columns:
            acidentes_br = self.df['br'].value_counts().head(10)
            print(f"\n🚨 TOP 10 BRs COM MAIS ACIDENTES:")
            for i, (br, qtd) in enumerate(acidentes_br.items(), 1):
                print(f"   {i:2d}. BR-{br}: {qtd:,} acidentes")
        
        # Análise por trecho específico (BR + KM)
        if 'br' in self.df.columns and 'km' in self.df.columns:
            # Agrupa por BR e faixa de KM (a cada 10km)
            self.df['km_faixa'] = (self.df['km'] // 10) * 10
            trechos_perigosos = self.df.groupby(['br', 'km_faixa']).size().reset_index(name='acidentes')
            trechos_perigosos = trechos_perigosos.nlargest(10, 'acidentes')
            
            print(f"\n🚨 TOP 10 TRECHOS MAIS PERIGOSOS:")
            for i, row in trechos_perigosos.iterrows():
                br = row['br']
                km_inicio = row['km_faixa']
                km_fim = km_inicio + 10
                acidentes = row['acidentes']
                print(f"   {i+1:2d}. BR-{br} (Km {km_inicio:.0f}-{km_fim:.0f}): {acidentes:,} acidentes")
        
        # Visualização dos trechos perigosos
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Trechos Mais Perigosos', fontsize=16, fontweight='bold')
        
        # Gráfico BRs
        if 'br' in self.df.columns:
            acidentes_br.plot(kind='barh', ax=axes[0], color='orange')
            axes[0].set_title('Top 10 BRs com Mais Acidentes')
            axes[0].set_xlabel('Número de Acidentes')
            axes[0].set_ylabel('BR')
        
        # Gráfico por estado
        if 'uf' in self.df.columns:
            acidentes_uf = self.df['uf'].value_calls().head(10)
            acidentes_uf.plot(kind='barh', ax=axes[1], color='red')
            axes[1].set_title('Top 10 Estados com Mais Acidentes')
            axes[1].set_xlabel('Número de Acidentes')
            axes[1].set_ylabel('Estado')
        
        plt.tight_layout()
        plt.savefig('trechos_perigosos.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: trechos_perigosos.png")
    
    def analise_condicoes_fatais(self):
        """Analisa condições que levam a acidentes fatais"""
        print("\n" + "="*70)
        print("☠️ ANÁLISE DE CONDIÇÕES QUE LEVAM A ACIDENTES FATAIS")
        print("="*70)
        
        # Filtra apenas acidentes com mortos
        acidentes_fatais = self.df[self.df['mortos'] > 0] if 'mortos' in self.df.columns else pd.DataFrame()
        
        if len(acidentes_fatais) == 0:
            print("❌ Não foi possível identificar acidentes fatais nos dados")
            return
        
        total_acidentes = len(self.df)
        total_fatais = len(acidentes_fatais)
        taxa_fatalidade = (total_fatais / total_acidentes) * 100
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Total de acidentes: {total_acidentes:,}")
        print(f"   • Acidentes fatais: {total_fatais:,}")
        print(f"   • Taxa de fatalidade: {taxa_fatalidade:.2f}%")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Acidentes Fatais', fontsize=16, fontweight='bold')
        
        # 1. Fatalidade por hora
        if 'hora' in self.df.columns:
            fatais_hora = acidentes_fatais.groupby('hora').size()
            total_hora = self.df.groupby('hora').size()
            taxa_hora = (fatais_hora / total_hora * 100).fillna(0)
            
            taxa_hora.plot(kind='bar', ax=axes[0,0], color='darkred')
            axes[0,0].set_title('Taxa de Fatalidade por Hora (%)')
            axes[0,0].set_xlabel('Hora')
            axes[0,0].set_ylabel('Taxa de Fatalidade (%)')
            axes[0,0].tick_params(axis='x', rotation=0)
            
            # Identifica horários mais fatais
            top_fatais_hora = taxa_hora.nlargest(3)
            print(f"\n🚨 HORÁRIOS COM MAIOR TAXA DE FATALIDADE:")
            for hora, taxa in top_fatais_hora.items():
                print(f"   • {hora:02d}:00h - {taxa:.1f}% de fatalidade")
        
        # 2. Fatalidade por condição meteorológica
        if 'condicao_metereologica' in self.df.columns:
            fatais_clima = acidentes_fatais['condicao_metereologica'].value_counts().head(5)
            fatais_clima.plot(kind='barh', ax=axes[0,1], color='purple')
            axes[0,1].set_title('Acidentes Fatais por Condição Meteorológica')
            axes[0,1].set_xlabel('Número de Acidentes Fatais')
        
        # 3. Fatalidade por tipo de pista
        if 'tipo_pista' in self.df.columns:
            fatais_pista = acidentes_fatais['tipo_pista'].value_counts().head(5)
            fatais_pista.plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
            axes[1,0].set_title('Acidentes Fatais por Tipo de Pista')
            axes[1,0].set_ylabel('')
        
        # 4. Fatalidade por fase do dia
        if 'fase_dia' in self.df.columns:
            fatais_fase = acidentes_fatais['fase_dia'].value_counts()
            fatais_fase.plot(kind='bar', ax=axes[1,1], color='navy')
            axes[1,1].set_title('Acidentes Fatais por Fase do Dia')
            axes[1,1].set_xlabel('Fase do Dia')
            axes[1,1].set_ylabel('Número de Acidentes Fatais')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('condicoes_fatais.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: condicoes_fatais.png")
    
    def analise_causas_principais(self):
        """Analisa as principais causas dos acidentes"""
        print("\n" + "="*70)
        print("🔍 ANÁLISE DAS PRINCIPAIS CAUSAS DOS ACIDENTES")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Principais Causas dos Acidentes', fontsize=16, fontweight='bold')
        
        # 1. Top causas gerais
        if 'causa_acidente' in self.df.columns:
            causas = self.df['causa_acidente'].value_counts().head(10)
            print(f"\n🚨 TOP 10 CAUSAS DE ACIDENTES:")
            for i, (causa, qtd) in enumerate(causas.items(), 1):
                print(f"   {i:2d}. {causa}: {qtd:,} acidentes")
            
            causas.plot(kind='barh', ax=axes[0,0], color='coral')
            axes[0,0].set_title('Top 10 Causas de Acidentes')
            axes[0,0].set_xlabel('Número de Acidentes')
        
        # 2. Tipos de acidentes
        if 'tipo_acidente' in self.df.columns:
            tipos = self.df['tipo_acidente'].value_counts().head(8)
            tipos.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
            axes[0,1].set_title('Distribuição por Tipo de Acidente')
            axes[0,1].set_ylabel('')
        
        # 3. Classificação dos acidentes
        if 'classificacao_acidente' in self.df.columns:
            classificacao = self.df['classificacao_acidente'].value_counts()
            classificacao.plot(kind='bar', ax=axes[1,0], color='lightgreen')
            axes[1,0].set_title('Classificação dos Acidentes')
            axes[1,0].set_xlabel('Classificação')
            axes[1,0].set_ylabel('Número de Acidentes')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Causas vs Gravidade
        if 'causa_acidente' in self.df.columns and 'gravidade_numerica' in self.df.columns:
            # Pega top 5 causas e analisa gravidade
            top_causas = self.df['causa_acidente'].value_counts().head(5).index
            df_top_causas = self.df[self.df['causa_acidente'].isin(top_causas)]
            
            causa_gravidade = df_top_causas.groupby(['causa_acidente', 'gravidade_numerica']).size().unstack(fill_value=0)
            causa_gravidade.plot(kind='bar', stacked=True, ax=axes[1,1])
            axes[1,1].set_title('Top 5 Causas vs Gravidade')
            axes[1,1].set_xlabel('Causa do Acidente')
            axes[1,1].set_ylabel('Número de Acidentes')
            axes[1,1].legend(['Sem Vítimas', 'Feridos Leves', 'Feridos Graves', 'Mortos'])
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('causas_acidentes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: causas_acidentes.png")
    
    def analise_veiculos_envolvidos(self):
        """Analisa os tipos de veículos mais envolvidos em acidentes"""
        print("\n" + "="*70)
        print("🚗 ANÁLISE DE VEÍCULOS ENVOLVIDOS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Veículos Envolvidos', fontsize=16, fontweight='bold')
        
        # 1. Tipos de veículos mais envolvidos
        if 'tipo_veiculo' in self.df.columns:
            veiculos = self.df['tipo_veiculo'].value_counts().head(10)
            print(f"\n🚨 TOP 10 TIPOS DE VEÍCULOS EM ACIDENTES:")
            for i, (veiculo, qtd) in enumerate(veiculos.items(), 1):
                print(f"   {i:2d}. {veiculo}: {qtd:,} acidentes")
            
            veiculos.plot(kind='barh', ax=axes[0,0], color='steelblue')
            axes[0,0].set_title('Top 10 Tipos de Veículos')
            axes[0,0].set_xlabel('Número de Acidentes')
        
        # 2. Idade dos veículos
        if 'ano_fabricacao_veiculo' in self.df.columns and 'ano_arquivo' in self.df.columns:
            # Calcula idade do veículo
            self.df['idade_veiculo'] = self.df['ano_arquivo'] - self.df['ano_fabricacao_veiculo']
            # Remove idades negativas ou muito altas (dados inconsistentes)
            idades_validas = self.df[(self.df['idade_veiculo'] >= 0) & (self.df['idade_veiculo'] <= 50)]
            
            if len(idades_validas) > 0:
                idades_validas['idade_veiculo'].hist(bins=20, ax=axes[0,1], color='orange', alpha=0.7)
                axes[0,1].set_title('Distribuição da Idade dos Veículos')
                axes[0,1].set_xlabel('Idade do Veículo (anos)')
                axes[0,1].set_ylabel('Número de Acidentes')
                
                idade_media = idades_validas['idade_veiculo'].mean()
                print(f"\n📊 IDADE MÉDIA DOS VEÍCULOS: {idade_media:.1f} anos")
        
        # 3. Veículos vs Gravidade
        if 'tipo_veiculo' in self.df.columns and 'gravidade_numerica' in self.df.columns:
            top_veiculos = self.df['tipo_veiculo'].value_counts().head(5).index
            df_top_veiculos = self.df[self.df['tipo_veiculo'].isin(top_veiculos)]
            
            veiculo_gravidade = df_top_veiculos.groupby(['tipo_veiculo', 'gravidade_numerica']).size().unstack(fill_value=0)
            veiculo_gravidade_pct = veiculo_gravidade.div(veiculo_gravidade.sum(axis=1), axis=0) * 100
            
            veiculo_gravidade_pct.plot(kind='bar', stacked=True, ax=axes[1,0])
            axes[1,0].set_title('Top 5 Veículos vs Gravidade (%)')
            axes[1,0].set_xlabel('Tipo de Veículo')
            axes[1,0].set_ylabel('Percentual de Acidentes')
            axes[1,0].legend(['Sem Vítimas', 'Feridos Leves', 'Feridos Graves', 'Mortos'])
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Marcas mais envolvidas
        if 'marca' in self.df.columns:
            marcas = self.df['marca'].value_counts().head(8)
            marcas.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
            axes[1,1].set_title('Top 8 Marcas em Acidentes')
            axes[1,1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('veiculos_acidentes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: veiculos_acidentes.png")
    
    def analise_perfil_vitimas(self):
        """Analisa o perfil demográfico das vítimas"""
        print("\n" + "="*70)
        print("👥 ANÁLISE DO PERFIL DAS VÍTIMAS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Perfil Demográfico das Vítimas', fontsize=16, fontweight='bold')
        
        # 1. Distribuição por sexo
        if 'sexo' in self.df.columns:
            sexo_dist = self.df['sexo'].value_counts()
            sexo_dist.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', colors=['lightblue', 'pink'])
            axes[0,0].set_title('Distribuição por Sexo')
            axes[0,0].set_ylabel('')
            
            print(f"\n👥 DISTRIBUIÇÃO POR SEXO:")
            for sexo, qtd in sexo_dist.items():
                pct = (qtd / sexo_dist.sum()) * 100
                print(f"   • {sexo}: {qtd:,} ({pct:.1f}%)")
        
        # 2. Distribuição por faixa etária
        if 'idade' in self.df.columns:
            # Remove idades inválidas
            idades_validas = self.df[(self.df['idade'] >= 0) & (self.df['idade'] <= 100)]
            
            if len(idades_validas) > 0:
                idades_validas['idade'].hist(bins=20, ax=axes[0,1], color='lightgreen', alpha=0.7)
                axes[0,1].set_title('Distribuição por Idade')
                axes[0,1].set_xlabel('Idade')
                axes[0,1].set_ylabel('Número de Pessoas')
                
                idade_media = idades_validas['idade'].mean()
                idade_mediana = idades_validas['idade'].median()
                print(f"\n📊 ESTATÍSTICAS DE IDADE:")
                print(f"   • Idade média: {idade_media:.1f} anos")
                print(f"   • Idade mediana: {idade_mediana:.1f} anos")
        
        # 3. Tipo de envolvimento
        if 'tipo_envolvido' in self.df.columns:
            envolvimento = self.df['tipo_envolvido'].value_counts()
            envolvimento.plot(kind='bar', ax=axes[1,0], color='coral')
            axes[1,0].set_title('Tipo de Envolvimento')
            axes[1,0].set_xlabel('Tipo')
            axes[1,0].set_ylabel('Número de Pessoas')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Estado físico das vítimas
        if 'estado_fisico' in self.df.columns:
            estado_fisico = self.df['estado_fisico'].value_counts()
            estado_fisico.plot(kind='barh', ax=axes[1,1], color='gold')
            axes[1,1].set_title('Estado Físico das Vítimas')
            axes[1,1].set_xlabel('Número de Pessoas')
        
        plt.tight_layout()
        plt.savefig('perfil_vitimas.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📁 Gráfico salvo: perfil_vitimas.png")
    
    def relatorio_executivo(self):
        """Gera um relatório executivo com os principais insights"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO EXECUTIVO - PRINCIPAIS INSIGHTS")
        print("="*80)
        
        total_registros = len(self.df)
        total_acidentes = self.df['id'].nunique() if 'id' in self.df.columns else 0
        
        print(f"\n📊 RESUMO GERAL:")
        print(f"   • Total de pessoas envolvidas: {total_registros:,}")
        print(f"   • Total de acidentes únicos: {total_acidentes:,}")
        print(f"   • Período analisado: 2020-2025")
        
        # Principais insights
        insights = []
        
        # Horário mais perigoso
        if 'hora' in self.df.columns:
            hora_perigosa = self.df.groupby('hora').size().idxmax()
            qtd_hora = self.df.groupby('hora').size().max()
            insights.append(f"Horário mais perigoso: {hora_perigosa:02d}:00h ({qtd_hora:,} acidentes)")
        
        # BR mais perigosa
        if 'br' in self.df.columns:
            br_perigosa = self.df['br'].value_counts().index[0]
            qtd_br = self.df['br'].value_counts().iloc[0]
            insights.append(f"BR mais perigosa: BR-{br_perigosa} ({qtd_br:,} acidentes)")
        
        # Principal causa
        if 'causa_acidente' in self.df.columns:
            causa_principal = self.df['causa_acidente'].value_counts().index[0]
            qtd_causa = self.df['causa_acidente'].value_counts().iloc[0]
            insights.append(f"Principal causa: {causa_principal} ({qtd_causa:,} acidentes)")
        
        # Taxa de fatalidade
        if 'mortos' in self.df.columns:
            acidentes_fatais = (self.df['mortos'] > 0).sum()
            taxa_fatalidade = (acidentes_fatais / total_registros) * 100
            insights.append(f"Taxa de fatalidade: {taxa_fatalidade:.2f}%")
        
        print(f"\n🎯 PRINCIPAIS INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        arquivos = [
            'analise_temporal_acidentes.png',
            'trechos_perigosos.png', 
            'condicoes_fatais.png',
            'causas_acidentes.png',
            'veiculos_acidentes.png',
            'perfil_vitimas.png'
        ]
        for arquivo in arquivos:
            print(f"   • {arquivo}")
        
        print(f"\n💡 RECOMENDAÇÕES:")
        print(f"   • Intensificar fiscalização nos horários de maior risco")
        print(f"   • Implementar melhorias nos trechos mais perigosos")
        print(f"   • Campanhas educativas focadas nas principais causas")
        print(f"   • Monitoramento especial em condições meteorológicas adversas")
    
    def executar_analise_completa(self):
        """Executa todas as análises"""
        print("🚗 ANÁLISE EXPLORATÓRIA COMPLETA - ACIDENTES DE TRÂNSITO")
        print("="*80)
        
        if not self.carregar_dados():
            return
        
        # Executa todas as análises
        self.analise_temporal_detalhada()
        self.analise_trechos_perigosos()
        self.analise_condicoes_fatais()
        self.analise_causas_principais()
        self.analise_veiculos_envolvidos()
        self.analise_perfil_vitimas()
        self.relatorio_executivo()
        
        print(f"\n✅ Análise exploratória completa finalizada!")

def main():
    """Função principal"""
    analisador = AnalisadorPadroesAcidentes()
    analisador.executar_analise_completa()

if __name__ == "__main__":
    main()
