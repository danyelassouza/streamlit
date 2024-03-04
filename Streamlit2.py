import streamlit as st

import numpy as np
import datetime
from datetime import datetime
from bcb import Expectativas
from bcb import sgs
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
from plotnine import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from functools import reduce


#%% Variáveis

#Cores para gráficos

colors = {
    'blue':'#282F6B',
    'red' :'#B22200',
    'green':'#224F20',
    'yellow':'#EACE3F',
    'gray':'#666666',
    'orange':'#B35C1E',
    'purple':'#5F487C' 
}


#%% Funções para biblioteca bcb
#Cria uma função para a equação de fisher

def fisher(juros, inflacao):
    """ 
    Calcula a taxa de jjuros real neutra usando a equação fisher.

    Args:
        juros (float): a taxa de juros nominal em porcentagem (%).
        infacao (float): A taxa de inflação em porcentagem (%).

    Return:
        float: A taxa de juros real em porcentagem (%).
         
    Raises:
        TypeError: Se os argumentos 'juros' e 'inflacao' nao forem do tipo 'float'.

    Exemplo:
        >>> fisher(10,3)
        6.796116504854364    
    
    """

    juros = ((((1+(juros / 100)) / (1 + inflacao / 100 ))) - 1) * 100
    return juros


#Cria funçao para calcular a data de referência a partir da data de observação
def reference_date(date: str, n):
    """ 
    Calcula a data de referencia adicionando 3 anos a uma data de observação

    Args:
        date (str): Uma string que representa uma data no formato 'YYYY-MM-DD'.

    Return:
        List[str]: Uma lista de strings com a data de referencia no formato 'YYYY'.

    Raises:
        TypeError: Se o argumento 'date' não for uma string.

    Exemplo:
        >>> reference_date('2022-01-01', n+3)
        ['2025']
        
    """
    years = pd.DatetimeIndex(date).year.values + n # Calcula n anos a frente
    years = years.tolist()
    years = [str(i) for i in years]
    return years

#Cria a função para coletar os dados de títulos do tesouro direto

def tesouro_direto():
    url = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv'
    df = pd.read_csv(url, sep = ';', decimal = ',')
    df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], dayfirst =True)
    df['Data Base'] = pd.to_datetime(df['Data Base'], dayfirst =True)
    multi_indice = pd.MultiIndex.from_frame(df.iloc[:, :1])
    df = df.set_index(multi_indice).iloc[:, 1:]
    df = df.rename(columns = {'Data Vencimento' : 'matur',
                              'Data Base' : 'data',
                              'Taxa Compra Manha' : 'taxa_bid'})
    return df

#%% Aquisição e tratamento de dados

@st.cache_data
def carregar_dados():
#instância a classe de Expectativa
    em = Expectativas()

    #Começa com a API das Expectativas de Mercado Anuais
    exp_anual = em.get_endpoint('ExpectativasMercadoAnuais')

    #Importa as expectativas do IPCA anuais e realiza os filtros
    ipca_e_raw=(
        exp_anual.query()
        .filter(exp_anual.Indicador == 'IPCA')
        .filter(exp_anual.baseCalculo == 0 )
        .select(exp_anual.Data, exp_anual.Mediana, exp_anual.DataReferencia)
        .collect()
    )

    # Realiza o filtro para a data de referência 1 ano a frente das obs.
    ipca_e_t1 = ipca_e_raw[(
                ipca_e_raw
                .DataReferencia == reference_date(ipca_e_raw['Data'], n = 1)
                )]

    # Realiza o filtro para a data de referência 4 anos a frente das obs.
    ipca_e_t4 = ipca_e_raw[(
                ipca_e_raw
                .DataReferencia == reference_date(ipca_e_raw['Data'], n = 4)
                )]

    #Renomeia as colunas e Mensaliza

    ipca_e_t1 = (
                ipca_e_t1
                .rename(columns =                                       #mudança de nome da coluna
                                        {'Data' : 'date',
                                        'Mediana' : 'ipca_e_t1'})
                .drop(['DataReferencia'], axis = 1)                             #retira coluna que nao precisa
                .assign(date = lambda x: pd.PeriodIndex(x['date'], freq ='M'))  #nova coluna de data, frequencia mensal
                .loc[:, ['date', 'ipca_e_t1']]                                  #mantem 2 colunas
                .groupby(by = 'date')                                           #agrupamento por data
                .agg(ipca_e_t1 = ('ipca_e_t1', 'mean'))                         #calculo agregado da media
                .reset_index()
                .assign(date = lambda x : x.date.dt.to_timestamp())             #coluna em datetime

    )


    #Renomeia as colunas e Mensaliza

    ipca_e_t4 = (
                ipca_e_t4
                .rename(columns=
                                        {'Data' : 'date',
                                        'Mediana' : 'ipca_e_t4'})
                .drop(['DataReferencia'], axis = 1)
                .assign(date = lambda x: pd.PeriodIndex(x['date'], freq ='M'))
                .loc[:, ['date', 'ipca_e_t4']]
                .groupby(by = 'date')
                .agg(ipca_e_t4 = ('ipca_e_t4', 'mean'))
                .reset_index()
                .assign(date = lambda x : x.date.dt.to_timestamp())

    )

    #Importa as expectativas da Selic anuais e realiza os filtros
    selic_e_raw=(
        exp_anual.query()
        .filter(exp_anual.Indicador == 'Selic')
        .filter(exp_anual.baseCalculo == 0 )
        .select(exp_anual.Data, exp_anual.Mediana, exp_anual.DataReferencia)
        .collect()
    )

    #Realiza o filtro para data referencia 1 ano a frente das obs.
    selic_e_t1 = selic_e_raw[(
                                    selic_e_raw
                                    .DataReferencia == reference_date(selic_e_raw['Data'], n = 1)
                                    )]

    #Realiza o filtro para data referencia 4 ano a frente das obs.
    selic_e_t4 = selic_e_raw[(
                                    selic_e_raw
                                    .DataReferencia == reference_date(selic_e_raw['Data'], n = 4)
                                    )]

    #Renomeia as colunas
    selic_e_t1 = (
                selic_e_t1
                .rename(columns =                                       #mudança de nome da coluna
                                        {'Data' : 'date',
                                        'Mediana' : 'selic_e_t1'})
                .drop(['DataReferencia'], axis = 1)                             #retira coluna que nao precisa
                .assign(date = lambda x: pd.PeriodIndex(x['date'], freq ='M'))  #nova coluna de data, frequencia mensal
                .loc[:, ['date', 'selic_e_t1']]                                  #mantem 2 colunas
                .groupby(by = 'date')                                           #agrupamento por data
                .agg(selic_e_t1 = ('selic_e_t1', 'mean'))                         #calculo agregado da media
                .reset_index()
                .assign(date = lambda x : x.date.dt.to_timestamp())             #coluna em datetime

    )

    #Renomeia as colunas
    selic_e_t4 = (
                selic_e_t4
                .rename(columns =                                       #mudança de nome da coluna
                                        {'Data' : 'date',
                                        'Mediana' : 'selic_e_t4'})
                .drop(['DataReferencia'], axis = 1)                             #retira coluna que nao precisa
                .assign(date = lambda x: pd.PeriodIndex(x['date'], freq ='M'))  #nova coluna de data, frequencia mensal
                .loc[:, ['date', 'selic_e_t4']]                                  #mantem 2 colunas
                .groupby(by = 'date')                                           #agrupamento por data
                .agg(selic_e_t4 = ('selic_e_t4', 'mean'))                         #calculo agregado da media
                .reset_index()
                .assign(date = lambda x : x.date.dt.to_timestamp())             #coluna em datetime

    )

    #Junta os dados em uma data frame
    proxy_neutro_t4=(
        pd.merge(
                left = ipca_e_t4,
                right = selic_e_t4,
                how = 'inner',
                on = 'date')
        .assign(neutro_t4 = lambda x: fisher(x.selic_e_t4, x.ipca_e_t4))
    )

    #Trimestraliza o juros neutro
    proxy_neutro_t4 = (
        proxy_neutro_t4
        .assign(date_quarter = lambda x: pd.PeriodIndex(x['date'], freq ='Q'))  #nova coluna de data, frequencia trimestral
        .loc[:, ['date_quarter', 'neutro_t4']]                                  #mantem 2 colunas
        .groupby(by = 'date_quarter')                                           #agrupamento por data
        .agg(neutro = ('neutro_t4', 'mean'))                         #calculo agregado da media
        .reset_index()
                
    )

    #Junta os dados em uma data frame
    proxy_neutro_t1=(
        pd.merge(
                left = ipca_e_t1,
                right = selic_e_t1,
                how = 'inner',
                on = 'date')
        .assign(neutro_t1 = lambda x: fisher(x.selic_e_t1, x.ipca_e_t1))
    )

    #Trimestraliza o juros neutro
    proxy_neutro_t1 = (
        proxy_neutro_t1
        .assign(date_quarter = lambda x: pd.PeriodIndex(x['date'], freq ='Q'))  #nova coluna de data, frequencia trimestral
        .loc[:, ['date_quarter', 'neutro_t1']]                                  #mantem 2 colunas
        .groupby(by = 'date_quarter')                                           #agrupamento por data
        .agg(proxy_neutro_t1= ('neutro_t1', 'mean'))                         #calculo agregado da media
                
    )


    #Calcula o filtro HP
    filtro_hp = sm.tsa.filters.hpfilter(x = proxy_neutro_t1['proxy_neutro_t1'], lamb = 1600)

    #Salva tendência calculada
    proxy_neutro_hp = pd.DataFrame(filtro_hp[1]).reset_index() #posição 1 é tendência (0-ciclo)

    #Junta os dados e transforma de wide para long, deixa de ser largo e se torna longo
    #Junta os dados e transforma de wide para long, deixa de ser largo e se torna longo
    proxy1 = (
        pd.merge(proxy_neutro_t4, proxy_neutro_hp, on = 'date_quarter')
        .assign(date_quarter = lambda x:x.date_quarter.dt.to_timestamp())
        .rename(columns = {'proxy_neutro_t1_trend' : 'Selic real esperada em 1 ano, filtro HP (Focus)', 
                        'neutro' : 'Selic real esperada em 4 ano (Focus)' })
        .melt(id_vars = ['date_quarter'], var_name = 'proxy', value_name = 'values')
    )
    
    
    # Coleta e tratamento do Hiato do Produto do BC

    hiato = (
        pd.read_excel(
            "https://www.bcb.gov.br/content/ri/relatorioinflacao/202306/ri202306anp.xlsx",
            sheet_name = "Graf 2.2.4",
            skiprows = 8
        )
        .assign(date_quarter = lambda x: pd.PeriodIndex(x['Trimestre'], freq = 'Q'),
                hiato_bcb = lambda x: x.Hiato.astype(float))
        .loc[:, ['date_quarter', 'hiato_bcb']]
        .set_index('date_quarter')
    )

#Coleta de dados do PIB
    pib = (
        sgs.get({'pib' : 22109}) #coleta
        .assign(date_quarter = lambda x: pd.PeriodIndex(x.index, freq = 'Q')) #transforma a coluna de data
        .set_index('date_quarter') # add coluna de data no indice
    )

 #Calcula o filtro HP
    filtro_hp = sm.tsa.filters.hpfilter(x = pib['pib'], lamb = 1600)

    #Salva tendencia calculada
    potencial_hp = filtro_hp[1] #posição 1 é a tendencia (0-ciclo);

    hiato['hiato_hp'] = (pib["pib"] / potencial_hp - 1) * 100

#Regressão linear aplicado a especificação de Hamilton
    reg3 = smf.ols(
        formula = "pib ~ pib.shift(8) + pib.shift(9) + pib.shift(10) + pib.shift(11)", #especificação do modelo no formato da formula
        data = pib #fonte de dados
        ).fit() #estima o modelo

    #Salva a tendencia estimada
    potencial_h = reg3.predict() #extrai os valores estimados

    #Adiciona 11 NaNs no inicio da série para corresponder ao tamanho da serie do PIB
    potencial_h = np.append([np.nan]*11, potencial_h)

    #calcula o hiato
    hiato['hiato_hamilton'] = (pib["pib"] / potencial_h - 1) * 100

#Junta o Hiato com a proxy R_t4 (Taxa de alta frequência)
    taxa_freq = proxy_neutro_t4.set_index('date_quarter').join(hiato, on = 'date_quarter').dropna()

#produz a taxa real de baixa frequência
    taxa_freq['r_n_bcb'] = (taxa_freq.neutro + (taxa_freq.hiato_bcb - 0.84 * (taxa_freq.hiato_bcb.shift(1))) / 0.75)
    taxa_freq['r_n_hp'] = (taxa_freq.neutro + (taxa_freq.hiato_hp - 0.84 * (taxa_freq.hiato_hp.shift(1))) / 0.75)
    taxa_freq['r_n_hamilton'] = (taxa_freq.neutro + (taxa_freq.hiato_hamilton - 0.84 * (taxa_freq.hiato_hamilton.shift(1))) / 0.75)

# Realiza o tratamento para o plot

    proxy2_af = (
        taxa_freq
        .reset_index()
        .drop(['hiato_bcb', 'hiato_hp', 'hiato_hamilton'], axis = 1)
        .assign(date_quarter = lambda x: x.date_quarter.dt.to_timestamp())
        .rename(columns = {'r_n_bcb' : 'Hiato BCB',
                        'r_n_hp' : 'Hiato Filtro HP',
                        'r_n_hamilton' : 'Hiato Filtro de Hamilton',
                        'neutro' : 'Selic real esperada em 4 anos (Focus)'})
        .melt(id_vars = ['date_quarter'], var_name = 'proxy', value_name = 'values')
    )

        #retire os dados faltantes
    taxa_freq.dropna(inplace = True)

    #Calcula o filtro HP para cada medida de taxa real de alta frequência
    taxa_freq['r_n_bcb_bf'] = sm.tsa.filters.hpfilter(x = taxa_freq['r_n_bcb'], lamb = 1600)[1] #Hiato BCB
    taxa_freq['r_n_hp_bf'] = sm.tsa.filters.hpfilter(x = taxa_freq['r_n_hp'], lamb = 1600)[1] #Hiato HP
    taxa_freq['r_n_hamilton_bf'] = sm.tsa.filters.hpfilter(x = taxa_freq['r_n_hamilton'], lamb = 1600)[1] #Hiato Hamilton
    taxa_freq['neutro_bf'] = sm.tsa.filters.hpfilter(x = taxa_freq['neutro'], lamb = 1600)[1] #Selic t4

    proxy2_bf = (
        taxa_freq
        .reset_index()
        .drop(['hiato_bcb', 'hiato_hp', 'hiato_hamilton', 'r_n_bcb', 'r_n_hp', 'r_n_hamilton', 'neutro'], axis = 1)
        .assign(date_quarter = lambda x: x.date_quarter.dt.to_timestamp())
        .rename(columns = {'r_n_bcb_bf' : 'Hiato BCB',
                        'r_n_hp_bf' : 'Hiato Filtro HP',
                        'r_n_hamilton_bf' : 'Hiato Filtro de Hamilton',
                        'neutro_bf' : 'Selic real esperada em 4 anos (Focus)'})
        .melt(id_vars = ['date_quarter'], var_name = 'proxy', value_name = 'values')
    )

    #importa os dados
    titulos = tesouro_direto()

    #Filtro para a NTN-B
    ntnb = titulos.loc['Tesouro IPCA+ com Juros Semestrais']

    #Cria coluna de ano inteiro para a maturidade
    ntnb['matur_year'] = ntnb.matur.dt.year

    #Filtro NTN-B 35
    ntnb_t35 = ntnb[(
                    ntnb
                    .matur_year == 2035)
                ]

    #Filtro NTN-B 45
    ntnb_t45 = ntnb[(
                    ntnb
                    .matur_year == 2045)
                ]
    
    #Trata os dados e trimestraliza
    ntnb_t35 = (
                ntnb_t35
                .reset_index(drop = True)
                .assign(date_quarter = lambda x: pd.PeriodIndex(x['data'], freq ='Q'))  #nova coluna de data, frequencia trimestral
                .loc[:, ['date_quarter', 'taxa_bid', 'matur_year']]                                  #mantem 2 colunas
                .groupby(by = 'date_quarter')                                           #agrupamento por data
                .agg(taxa_bid_35 = ('taxa_bid', 'mean'))                         #calculo agregado da media
                .reset_index()
                )

    #Trata os dados e trimestraliza
    ntnb_t45 = (
                ntnb_t45
                .reset_index(drop = True)
                .assign(date_quarter = lambda x: pd.PeriodIndex(x['data'], freq ='Q'))  #nova coluna de data, frequencia trimestral
                .loc[:, ['date_quarter', 'taxa_bid', 'matur_year']]                                  #mantem 3 colunas
                .groupby(by = 'date_quarter')                                           #agrupamento por data
                .agg(taxa_bid_45 = ('taxa_bid', 'mean'))                         #calculo agregado da media
                .reset_index()
    )

    #Lista de dfs para juntar
    dfs = [proxy_neutro_t4, ntnb_t35, ntnb_t45]

    #Junta as dfs
    termos = reduce(lambda left, right: pd.merge(right, left, on = 'date_quarter'), dfs).dropna()

    #Aplica o calculo da Taxa real de mercado descontando prêmio a termo
    termos['taxa_real_35'] = (termos.taxa_bid_35 - (termos.taxa_bid_35 - termos.neutro).mean())
    termos['taxa_real_45'] = (termos.taxa_bid_45 - (termos.taxa_bid_45 - termos.neutro).mean())

    #Realiza tratamento para o plot
    proxy3 = (
        termos
        .drop(['taxa_bid_45', 'taxa_bid_35', 'neutro'], axis = 1)
        .assign(date_quarter = lambda x: x.date_quarter.dt.to_timestamp())
        .rename(columns = {'taxa_bid_35' : 'NTN-B 35',
                        'taxa_bid_45' : 'NYN-B 45'})
        .melt(id_vars = ['date_quarter'], var_name = 'proxy', value_name = 'values')
    )

    #taxa real de mercado de longo prazo com filtro HP

    #Define o indice
    ntnb_t45.set_index('date_quarter', inplace = True)

    #Calcula filtro HP
    filtro_hp_taxa = sm.tsa.filters.hpfilter(x = ntnb_t45['taxa_bid_45'], lamb = 1600)

    #Salva tendência calculada
    proxy4 = pd.DataFrame(filtro_hp_taxa[1])

    #reinicia o indice e transforma a coluna date_quarter para datetime
    proxy4 = proxy4.reset_index().assign(date_quarter = lambda x: x.date_quarter.dt.to_timestamp())


    return proxy1,proxy2_af,proxy2_bf,proxy3,proxy4
#%% Streamlit


st.set_page_config(page_title="Juros Neutro",layout="wide")

dados = carregar_dados()

with st.container():
    st.subheader("Selic")
    st.line_chart(dados[0], x="date_quarter", y="values", color="proxy")

 
with st.container():
    st.subheader("Taxas neutras de alta frequência extraídas de hiato e taxa real 4 anos à frente Focus")
    st.line_chart(dados[1], x="date_quarter", y="values", color="proxy")

with st.container():
    st.subheader("Taxas neutras de baixa frequência extraídas de hiato e taxa real 4 anos à frente FocusSelic")
    st.line_chart(dados[2], x="date_quarter", y="values", color="proxy")

    
with st.container():
    st.subheader("Taxas reais NTN-B descontando o prêmio a termo")
    st.line_chart(dados[3], x="date_quarter", y="values", color="proxy")

with st.container():
    st.subheader("Juro neutro brasileiro a partir da NTN-B 2045 com Filtro HP")
    st.line_chart(dados[4], x="date_quarter", y="taxa_bid_45_trend", color=list(colors.values())[1])