import pandas as                      pd
import numpy as                       np
import seaborn as                     sns
import matplotlib.pyplot as           plt
import plotly.express as              px
import                                warnings
import streamlit as                   st

from time import                      sleep
from matplotlib import                pylab
from sklearn.cluster import           KMeans
from sklearn.decomposition import     PCA
from sklearn.model_selection import   train_test_split
from scipy.spatial.distance import    cdist, pdist
from sklearn.metrics import           silhouette_score
from sklearn.preprocessing import     MinMaxScaler
from streamlit_option_menu import     option_menu

custom_params = {"axes.spines.right":False,"axes.spines.top":False}
sns.set_theme(style='ticks',rc=custom_params)

@st.cache_data
def load_data(file_data):
    try:
        if file_data.name.endswith('.txt'):
            return pd.read_csv(file_data, delimiter=';')  
        elif file_data.name.endswith('.csv'):
            return pd.read_csv(file_data, delimiter=';')
        else:
            return pd.read_excel(file_data)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None


def treatment(df):
    df['Voltage'] = df['Voltage'].astype(dtype = 'float64')
    df['Global_active_power'] = df['Global_active_power'].astype(dtype = 'float64')
    df['Global_reactive_power'] = df['Global_reactive_power'].astype(dtype = 'float64')
    df['Global_intensity'] = df['Global_intensity'].astype(dtype = 'float64')
    df['Sub_metering_1'] = df['Sub_metering_1'].astype(dtype = 'float64')
    df['Sub_metering_2'] = df['Sub_metering_2'].astype(dtype = 'float64')
    df['Sub_metering_3'] = df['Sub_metering_3'].astype(dtype = 'float64')
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.drop(columns=['Time','Date'],inplace=True)
    df = df.reset_index(drop=True)
    return df

def format_blue(text:str):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h2 style="color: lightblue;">{text}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def main():
    st.set_page_config(page_title="Clustering",
                       page_icon='central.jfif',
                       layout='wide',
                       initial_sidebar_state='expanded')
    format_blue("Projeto Machine Learning - Clusterização")
    st.sidebar.image('central.jfif')
    
    st.markdown("---")

    st.markdown("""
Nosso projeto será baseado na metodologia CRISP-DM, que é a abreviação de Cross Industry Standard Process for Data Mining, que pode ser traduzido como Processo Padrão Inter-Indústrias para Mineração de Dados. É um modelo de processo de mineração de dados que descreve abordagens comumente usadas por especialistas em mineração de dados para atacar problemas. É formada por 6 etapas bem definidas:

1. **Business Understanding**
2. **Data Understanding**
3. **Data Preparation**
4. **Modeling**
5. **Evaluation**
6. **Deployment**
""")
    st.write('---')
    data_file_1 = st.sidebar.file_uploader("", type=['csv', 'xlsx', 'txt'])

    st.sidebar.write(f"O link do arquivo dos dados se encontra em: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption ")
    st.sidebar.markdown("---")

    if data_file_1 is not None:
        try:
            dataset = load_data(data_file_1)
            #df = treatment(df)
        except Exception as e:
            st.sidebar.error(f"Error loading the file: {e}")

        with st.sidebar:
            selected_page = option_menu(
                menu_title="Navigation",
                options=["Business Understanding", "Data Understanding", "Data Preparation", 
                        "Modeling", "Evaluation"],
                icons=["play-circle", "bar-chart-line",  "shuffle", "hourglass", "shield"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                key="navigation_menu"
            )
        if selected_page == "Business Understanding":
            format_blue('Etapa 1 CRISP-DM: Entendimento do negócio')
            st.markdown("""
Como primeira etapa do CRISP-DM, vamos entender do que se trata o negócio e quais os objetivos.

Este é um problema de Consumo Individual de Energia Elétrica Doméstica. Nosso objetivo principal é através das análises exploratórias e também por meio de algoritmos de machine learning, conseguir agrupar e separar os indivíduos dessa base de dados para que possamos tomar atitudes baseadas em dados.

Então, como objetivo da modelagem é fazer uma clusterização, usaremos o KMeans para segmentar nossos clientes.
""")

        elif selected_page == 'Data Understanding':
            format_blue("Etapa 2 CRISP-DM: Entendimento dos dados")
            st.write('A segunda etapa é o entendimento dos dados. Foram fornecidas 9 variáveis. O significado de cada uma dessas variáveis se encontra na tabela.')
            st.write('''
**Dicionário de dados**

Os dados estão dispostos em uma tabela com uma linha para cada cliente, e uma coluna para cada variável armazenando as características desses clientes. Colocamos uma cópia o dicionário de dados (explicação dessas variáveis) abaixo neste notebook:
''')
            st.markdown("""
| Variable Name            | Description                                         | Tipo  |
| ------------------------ |:---------------------------------------------------:| -----:|
| Date| Data no formato dd/mm/aaaa|Categorical|
| Time| hora no formato hh:mm:ss |Categorical|
| Global_active_power|potência ativa média por minuto global doméstica (em quilowatts)|Categorical|
| Global_reactive_power| potência reativa média por minuto global doméstica (em quilowatts) |Categorical|
| Voltage|voltagem média por minuto (em volts)  | Categorical|
| Global_intensity | intensidade de corrente média por minuto global doméstica (em amperes) | Categorical |
| Sub_metering_1 | ubmedição de energia nº 1 (em watt-hora de energia ativa). Corresponde à cozinha, contendo principalmente uma máquina de lavar louça, um forno e um micro-ondas (os fogões não são elétricos, mas movidos a gás)| Categorical |
| Sub_metering_2 | submedição de energia nº 2 (em watt-hora de energia ativa). Corresponde à lavanderia, contendo uma máquina de lavar, uma secadora, uma geladeira e uma luz. |Categorical|
| Sub_metering_3 | submedição de energia nº 3 (em watt-hora de energia ativa). Corresponde a um aquecedor de água elétrico e a um ar-condicionado. |Continuous|

""")
            st.write('---')
            st.write('#### Amostragem dos dados')
            st.write(dataset.head())
            st.write(f"Inicialmente temos {dataset.shape[0]} linhas e {dataset.shape[1]} colunas em nosso data set")
            st.write('#### Visualizando Informações dos dados')
            col1,col2 = st.columns(2)
            with col1:
                st.write("- **Schema**")
                st.write(dataset.dtypes)
            with col2: 
                st.write("- **Missings**")
                st.write((dataset.isna().sum()))
            st.write('---')
            st.write("""
#### Entendimento dos dados - Univariada
Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis. Nesta demonstração vamos ver a variável resposta e dois exemplos de univariada apenas. Mas sinta-se à vontade para tentar observar outras variáveis.
""")        
            st.write('---')
            dataset.dropna(inplace=True)
            df, amostra2 = train_test_split(dataset, train_size = .01)
            df = treatment(df)

            fig, ax = plt.subplots(figsize=(20, 15))  
            df.hist(bins=32, ax=ax, grid=False) 
            st.pyplot(fig)
            st.write('---')
            st.write("**Outra maneira de vizualizar, agora tendo acesso aos quartis e aos outliers**")

            fig = plt.figure()
            num_cols = df.select_dtypes(include='number').columns
            num_plots = len(num_cols)

            quant_col = df.select_dtypes(['int','float']).columns.tolist()

            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 20))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.boxplot(data=df, x=col,linecolor="#0b2d3e",palette="mako_r", linewidth=.7,flierprops={"marker": "x"}, ax=axes[i])
                axes[i].set_title(f'Histograma de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(fig)
            st.write('---')
            st.write("**Conferindo como se comportam ao longo do tempo**")
            fig = plt.figure()
            num_cols = df.select_dtypes(include='number').columns
            num_plots = len(num_cols)


            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 20))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.scatterplot(data=df, x="DateTime" ,y=col, ax=axes[i],palette="mako_r")
                axes[i].set_title(f'Histograma de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(fig)
            st.write('---')

            st.write("**Correlação**")
            fig=plt.figure()
            sns.heatmap(df.drop(columns=['DateTime']).corr(),vmin=-1,vmax=1,cmap="mako_r")
            st.pyplot(fig)

            st.write('---')
            
            st.write('''**Podemos observar as características presentes na distribuição**''')
            st.info('OBS: Esse plot pode demorar um pouco devido a quantidade de dados')
            ax = sns.pairplot(df,diag_kind = 'kde',palette="mako_r")
            st.pyplot(ax.figure)

        elif selected_page == "Data Preparation":
            dataset.dropna(inplace=True)
            df, amostra2 = train_test_split(dataset, train_size = .01)
            df = treatment(df)
            format_blue("Etapa 3 CRISP-DM: Preparação dos Dados")
            st.markdown("""
Nessa etapa realizamos tipicamente as seguintes operações com os dados:
- seleção
- limpeza
- construção
- integração
- formatação



Porém como foi supracitado na etapa anterior, já tratamos os dados, eliminando as variáveis vazias, pois eram em apenas uma coluna e representava cerca de 1% apenas dos dados. Então fizemos a divisão para pegarmos apenas uma fatia desses 2 milhões de dados que tinhamos, e fizemos a correção do `Schema` (schema é a relação da variável e o tipo de dado em que ela se encontra). O que nos resta para trabalhar com os dados é coloca-los numa escala favorável para o modelo e aplicar uma análise de componentes principais sobre eles.
""")
            st.write("**Então basicamente o que fizemos antres de plotar a análise exploratória foi isso:**")

            st.markdown("""
```python
df, amostra2 = train_test_split(dataset, train_size = .01)
                        
df['Voltage'] = df['Voltage'].astype(dtype = 'float64')
df['Global_active_power'] = df['Global_active_power'].astype(dtype = 'float64')
df['Global_reactive_power'] = df['Global_reactive_power'].astype(dtype = 'float64')
df['Global_intensity'] = df['Global_intensity'].astype(dtype = 'float64')
df['Sub_metering_1'] = df['Sub_metering_1'].astype(dtype = 'float64')
df['Sub_metering_2'] = df['Sub_metering_2'].astype(dtype = 'float64')
df['Sub_metering_3'] = df['Sub_metering_3'].astype(dtype = 'float64')

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df.drop(columns=['Time','Date'],inplace=True)
df = df.reset_index(drop=True)
""")
            st.write("Também foi usado os algoritmos de PCA e MinMaxScaler.")
            df_ml = df.drop(columns=['DateTime'])
            scaler = MinMaxScaler()
            df_scale = scaler.fit_transform(df_ml)
            pca = PCA()
            pca_ = pca.fit_transform(df_scale)

            name = [f"CP{x+1}" for x in list(range(df_ml.shape[1]))]
            pca_df = pd.DataFrame(data = pca_,columns=name)
            st.markdown("""
```python
df_ml = df.drop(columns=['DateTime'])
scaler = MinMaxScaler()
df_scale = scaler.fit_transform(df_ml)
                        
pca = PCA()
pca_ = pca.fit_transform(df_scale)

name = [f"CP{x+1}" for x in list(range(df_ml.shape[1]))]
pca_df = pd.DataFrame(data = pca_,columns=name)
pca_df                                           
""")
            st.write(pca_df)
            mydf= pd.Series(pca.explained_variance_ratio_.cumsum().tolist()).to_frame()
            mydf.rename(columns={0:'First'},inplace=True)
            ax = sns.lineplot(data=mydf,x=mydf.index+1,y='First')
            st.pyplot(ax.figure)
            pca = PCA(n_components=3)
            pca = pca.fit_transform(df_scale)
            st.write("Basicamente criamos um PCA e avaliamos quanto da variance queremos explicar.Nesse caso, vendo o gráfico eu optei por apenas 3 variáveis")
            st.markdown("""
```python
                        pca = PCA(n_components=3)
pca = pca.fit_transform(df_scale)

pca
""")
            st.write('---')
        elif selected_page == 'Modeling':

            dataset.dropna(inplace=True)
            df, amostra2 = train_test_split(dataset, train_size = .01)
            df = treatment(df)
            df_ml = df.drop(columns=['DateTime'])
            scaler = MinMaxScaler()
            df_scale = scaler.fit_transform(df_ml)
            pca = PCA()
            pca_ = pca.fit_transform(df_scale)
            pca = PCA(n_components=3)
            pca = pca.fit_transform(df_scale)

            format_blue('Etapa 4 Crisp-DM: Modelagem')
            st.write("""
Nessa etapa, realizaremos a construção do modelo de clusterização. Os passos típicos são:

Seleção da Técnica de Modelagem
Para esse projeto de segmentação de consumidores de energia, utilizaremos a técnica de K-Means, que é amplamente usada em problemas de clusterização. O K-Means é um algoritmo simples, porém poderoso, que agrupa indivíduos com base em características semelhantes, distribuindo-os em clusters ou grupos distintos. Isso nos permitirá identificar padrões de comportamento no consumo de energia entre os clientes, facilitando a compreensão de diferentes perfis de consumo.

Além disso, aplicamos PCA (Análise de Componentes Principais) para reduzir a dimensionalidade dos dados e garantir que o modelo de clusterização se beneficie de uma representação mais compacta e eficiente. Também usamos o MinMaxScaler para normalizar os dados e garantir que as variáveis estejam na mesma escala, o que é essencial para o desempenho do K-Means.
""")
            st.info("Pode demorar um pouco, pois o algoritmo de clusterização esta sendo criado. Aguarde um momento.")
            k_range = range(1,12)

            # Aplicando o modelo K-Means para cada valor de K (esta célula pode levar bastante tempo para ser executada)
            k_means_var = [KMeans(n_clusters = k).fit(pca) for k in k_range]

            # Ajustando o centróide do cluster para cada modelo
            centroids = [X.cluster_centers_ for X in k_means_var]

            # Calculando a distância euclidiana de cada ponto de dado para o centróide
            k_euclid = [cdist(pca, cent, 'euclidean') for cent in centroids]
            dist = [np.min(ke, axis = 1) for ke in k_euclid]

            # Soma dos quadrados das distâncias dentro do cluster
            soma_quadrados_intra_cluster = [sum(d**2) for d in dist]

            # Soma total dos quadrados
            soma_total = sum(pdist(pca)**2)/pca.shape[0]

            # Soma dos quadrados entre clusters
            soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster

            st.write('---')
            st.write("#### Curva de Elbow")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(k_range, soma_quadrados_inter_cluster/soma_total * 100, 'b*-')
            ax.set_ylim((0,100))
            plt.xlabel('Número de Clusters')
            plt.ylabel('Percentual de Variância Explicada')
            plt.title('Variância Explicada x Valor de K') 
            st.pyplot(ax.figure)  
            st.write('---')
            pca_df = pd.DataFrame(pca)
            
            st.write("#### Coeficiente de Silhuetas")
            silhuetas = []
            max_cluster = 10

            for n_clusters in range(2, max_cluster):
                km = KMeans(n_clusters=n_clusters).fit(pca)
                silhuetas.append(silhouette_score(pca, km.labels_))
                nomes = [f'grupo_{g}' for g in range(n_clusters)]
                pca_df['grupos_' + str(n_clusters)] = pd.Categorical(km.labels_, categories=range(n_clusters), ordered=True)
            
            fig, ax = plt.subplots(figsize=(20, 15))  
            pd.Series(silhuetas).plot() 
            st.pyplot(fig)


            st.write("*Algoritmo criado, vimos que o numero de clusterns que nos interessa é 4. Vamos para parte de Avaliação e descobrir se realmente foi uma boa escolha.*")
        elif selected_page == "Evaluation":
            dataset.dropna(inplace=True)
            df, amostra2 = train_test_split(dataset, train_size = .01)
            df = treatment(df)
            df_ml = df.drop(columns=['DateTime'])
            scaler = MinMaxScaler()
            df_scale = scaler.fit_transform(df_ml)
            pca = PCA()
            pca_ = pca.fit_transform(df_scale)
            pca = PCA(n_components=3)
            pca = pca.fit_transform(df_scale)
            modelo_v2 = KMeans(n_clusters = 4)
            modelo_v2.fit(pca)
            labels = modelo_v2.labels_
            silhouette_score(pca, labels, metric = 'euclidean')
            cluster_map = df.copy()
            cluster_map['Global_active_power'] = pd.to_numeric(cluster_map['Global_active_power'])
            cluster_map['cluster'] = modelo_v2.labels_
            cluster_map.drop(columns=['DateTime'],inplace=True)
            st.markdown("""
## Etapa 5 CRISP-DM: Avaliação dos resultados
Na fase de avaliação do CRISP-DM, o objetivo é entender o desempenho do modelo de clusterização aplicado e como ele contribui para o projeto. No nosso caso, usamos o algoritmo K-Means para agrupar os dados transformados pelas três componentes principais do PCA, testando diferentes números de clusters.

Para realizar uma boa avaliação de clusterização, usamos métricas como o Silhouette Score, que mede a coesão e separação dos clusters. Os resultados obtidos foram:

- Silhouette Score com K=8: 0.3670
- Silhouette Score com K=4: 0.4494

**Análise do Silhouette Score**
O Silhouette Score é uma métrica que varia de -1 a 1:

- Valores próximos de 1 indicam que os clusters estão bem separados e que os pontos estão fortemente agrupados dentro de seu próprio cluster.
- Valores próximos de 0 indicam que os clusters estão sobrepostos, com pontos muito próximos dos limites dos clusters.
- Valores negativos indicam que muitos pontos estão mal alocados, ou seja, dentro do cluster errado.
No nosso caso, o melhor resultado foi obtido com K=4, com um Silhouette Score de 0.4494, o que indica uma separação moderada entre os clusters.

**Comparação com a Curva de Elbow**

Além do Silhouette Score, analisamos a Curva de Elbow, que nos ajuda a visualizar a soma dos quadrados dentro e entre clusters. Esta curva nos permite verificar como a variância explicada aumenta à medida que o número de clusters aumenta. No entanto, após certo ponto, o ganho se torna marginal, e o número de clusters ideal é aquele que precede essa estabilização.

Com base nos nossos resultados:

A Curva de Elbow sugeriu que 4 clusters é um bom ponto de equilíbrio, uma vez que a variação explicada aumentou consideravelmente até esse valor, mas os ganhos foram reduzidos para valores maiores de K.

**Considerações Finais**
- K=4 foi a escolha ideal para nosso projeto, dado que ofereceu o melhor equilíbrio entre a variância explicada e a qualidade dos clusters medida pelo Silhouette Score.
- Apesar de K=8 ter mostrado uma separação mais detalhada, o score mais baixo (0.3670) indica que a qualidade dos clusters foi inferior a K=4, com uma menor coesão interna.
- No contexto de clusterização, o impacto no negócio se dará conforme o uso dos clusters para segmentar clientes ou eventos. Em um cenário futuro, seria necessário avaliar como essas segmentações impactam decisões estratégicas, como alocação de recursos, segmentação de marketing, ou identificação de padrões de comportamento.

**Próximos Passos:**
Validação do modelo: Testar a robustez dos clusters com novos dados.
Análise de impacto no negócio: Identificar como esses grupos podem auxiliar na tomada de decisões.
Para este projeto, os resultados sugerem que uma clusterização com 4 grupos é suficiente para capturar a variabilidade dos dados, fornecendo uma separação clara e coesa entre os diferentes segmentos.
""")
            st.write('---')
            pca_df = pd.DataFrame(pca)
            cluster_map[pca_df.iloc[:,:3].columns] = pca_df.iloc[:,:3]
            fig = px.scatter_3d(cluster_map, x=0, y=1, z=2,
                                color='cluster',template='plotly_dark',color_continuous_scale='Blues')
            st.plotly_chart(fig)
            st.write('---')
            st.info('Aguarde...')
            ax = sns.pairplot(cluster_map.drop(columns=[0,1,2]),diag_kind = 'kde',hue='cluster',palette="mako")
            st.pyplot(ax.figure)
            st.write('---')
            num_cols = cluster_map.drop(columns=[0,1,2]).select_dtypes(include='number').columns
            num_plots = len(num_cols)

            quant_col = cluster_map.select_dtypes(['int','float']).columns.tolist()

            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 20))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.boxplot(data=cluster_map, x=col,linecolor="#0b2d3e",palette="mako",hue='cluster',linewidth=.7,flierprops={"marker": "x"}, ax=axes[i])
                axes[i].set_title(f'Histograma de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(fig)
            st.write('---')

            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 15))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.histplot(data=cluster_map, x=col, hue='cluster',palette="mako" ,bins=30, ax=axes[i], kde=False,multiple="stack")
                axes[i].set_title(f'Histograma de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(fig)
            st.write('---')
            st.write("""
Tabelas Crosstab para Explorar Padrões entre Variáveis e Clusters
As tabelas crosstab nos ajudam a identificar padrões entre as variáveis categóricas e os grupos formados pelos clusters. Vamos começar criando uma tabela cruzada entre as variáveis categóricas e os clusters.

Exemplo de Tabela Crosstab
Podemos cruzar, por exemplo, os clusters com a variável Global_active_power e observar a distribuição entre os clusters:""")
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Global_active_power'], normalize='index'))
            st.write('---')
            cluster_map['Voltage_qcut'] = pd.qcut(cluster_map['Voltage'], q=4, labels=['Baixo', 'Médio', 'Alto', 'Muito Alto'])
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Voltage_qcut']))
            st.write('---')
            cluster_map['Global_active_power_qcut'] = pd.qcut(cluster_map['Global_active_power'], q=4, labels=['Baixo', 'Médio', 'Alto', 'Muito Alto'])
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Global_active_power_qcut']))

            st.write("""**Interpretação dos Resultados de Crosstab**
**Distribuição de Voltage por Cluster:**

- O cluster 0 tem uma distribuição razoavelmente uniforme entre as categorias de Voltage, com um número similar de observações nas categorias "Baixo", "Médio", "Alto", e "Muito Alto".
- O cluster 1 tem um número maior de observações na categoria "Baixo" e "Médio", mas significativamente menos em "Alto" e "Muito Alto".
- O cluster 2 tem uma concentração muito maior em "Alto" e "Muito Alto", sugerindo que os dados deste cluster tendem a ter voltagens mais altas.
- O cluster 3 tem a maioria das observações nas categorias "Baixo" e "Médio", com pouquíssimas em "Alto" e "Muito Alto".

**Distribuição de Global_active_power por Cluster:**

- O cluster 0 está mais equilibrado entre as categorias de Global_active_power.
- O cluster 1 tem uma grande quantidade de observações em "Alto" e "Muito Alto", sugerindo que os dados nesse cluster têm potências ativas mais altas.
- O cluster 2 é composto majoritariamente pela categoria "Baixo", com uma diminuição gradual nas outras categorias.
- O cluster 3 é formado inteiramente pela categoria "Muito Alto", o que pode indicar uma segmentação clara para altos valores de potência ativa.""")
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Sub_metering_1'], normalize='index'))
            st.write('---')
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Sub_metering_2'], normalize='index'))
            st.write('---')
            st.write(pd.crosstab(cluster_map['cluster'], cluster_map['Sub_metering_3'], normalize='index'))
            st.write("""As variáveis **Sub_metering_1**, **Sub_metering_2** e **Sub_metering_3** medem o consumo de energia em três diferentes categorias de eletrodomésticos ou zonas de uma casa. Os **clusters (0, 1, 2, 3)** representam grupos de padrões de consumo, criados por meio de um algoritmo de clusterização .

**O que significam as proporções:**

Cada célula nas tabelas mostra a proporção de observações dentro de cada cluster para um valor específico da variável Sub_metering.

**Por exemplo:**

**Tabela de Sub_metering_1 (Consumo em uma zona específica)**
- **Para o cluster 0:**
    - 89,77% das observações nesse cluster têm o valor 0.0 de consumo em Sub_metering_1, o que significa que a grande maioria das casas ou dispositivos nesse cluster não consomem energia na zona representada por Sub_metering_1.
    - 4,37% têm o valor 1.0, ou seja, uma pequena fração de consumidores nesse cluster tem um baixo consumo.
- **Para o cluster 3:**
    - 47,01% das observações têm o valor 0.0, ou seja, menos da metade dos consumidores nesse grupo não consome energia nessa zona.
    - 9,56% têm o valor 1.0, indicando que mais pessoas nesse cluster consomem energia comparado ao cluster 0.

**O que isso significa?**

- O cluster 0 representa um grupo onde a maioria não consome energia em Sub_metering_1. Isso pode representar, por exemplo, casas que não têm aparelhos ligados a essa zona.

- Já o cluster 3 tem uma distribuição mais dispersa, com uma proporção menor de consumo zero, indicando que há mais variação no consumo de energia nessa zona.

**Tabela de Sub_metering_2 (Consumo em outra zona específica)**

- **Para o cluster 0:**

    - 45,22% das casas têm consumo zero, indicando uma maior diversidade de padrões de consumo.
    - 33,74% consomem 1.0 unidade de energia, o que significa que uma boa parte dos consumidores nesse grupo tem um consumo relativamente baixo.
      
- **Para o cluster 2:**

    - 83,31% das observações têm consumo zero, sugerindo que a maioria das casas nesse cluster não consome energia nessa zona.
  
**O que isso significa?**

- O cluster 0 tem um consumo mais variado e não tão concentrado no valor zero, indicando que há um uso mais ativo da zona de Sub_metering_2.
- O cluster 2, por outro lado, possui um grande número de consumidores com consumo zero, o que pode representar um grupo de pessoas que quase não utilizam essa área da casa.
  
**Tabela de Sub_metering_3 (Consumo em uma terceira zona)**

- O cluster 2 tem 64,52% de consumidores com consumo zero, indicando que a maioria não consome energia nessa área.
- Já o cluster 3 tem apenas 8,76% com consumo zero, o que significa que nesse grupo, a maioria dos consumidores está usando ativamente a zona representada por Sub_metering_3.

**Conclusão prática:**
Essas tabelas nos mostram como diferentes grupos de consumidores (clusters) utilizam energia nas três zonas representadas por Sub_metering_1, Sub_metering_2 e Sub_metering_3. Os clusters ajudam a identificar padrões de comportamento:

- Alguns grupos podem ter um baixo consumo em todas as áreas (como o cluster 2).
- Outros grupos têm mais variabilidade, com uma distribuição maior entre diferentes níveis de consumo (como o cluster 3).
  
**Em resumo:**
As tabelas crosstab te mostram como cada grupo de consumidores (clusters) se comporta em termos de consumo de energia. O objetivo é identificar padrões, como:

- Qual cluster consome mais ou menos energia em certas zonas da casa?
- Como o consumo de energia é distribuído em cada grupo?
  
Essas informações podem ser usadas para entender o comportamento de consumo e, eventualmente, propor estratégias de otimização, como programas de eficiência energética, ajustes tarifários, ou manutenção específica para cada grupo.""")

main()