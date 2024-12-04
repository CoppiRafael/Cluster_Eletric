

# Projeto de Machine Learning: Clusterização de Clientes por Consumo de Energia

## Descrição
Este projeto utiliza técnicas de **clusterização** para analisar padrões de consumo de energia elétrica doméstica com base nos dados disponibilizados pelo [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption). O objetivo é agrupar consumidores por similaridade no consumo de energia, auxiliando na identificação de perfis de comportamento e facilitando decisões baseadas em dados.

## Metodologia
O projeto segue a metodologia **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, um processo padrão para mineração de dados estruturado em seis etapas:

1. **Entendimento do Negócio**  
   Definir objetivos e metas com base nas necessidades do negócio.
   
2. **Entendimento dos Dados**  
   Compreender as variáveis disponíveis e identificar sua relevância para o problema.
   
3. **Preparação dos Dados**  
   Realizar limpeza, transformação e normalização dos dados.
   
4. **Modelagem**  
   Implementar algoritmos de clusterização, como o K-Means, e técnicas como PCA (Análise de Componentes Principais) para otimização.
   
5. **Avaliação**  
   Analisar a qualidade dos clusters utilizando métricas como **Silhouette Score** e a **Curva de Elbow**.
   
6. **Implantação**  
   Criar um aplicativo interativo para visualização e uso dos resultados.

---

## Detalhamento do Projeto

### Dataset
O conjunto de dados contém informações sobre o consumo de energia em diferentes áreas residenciais.  
**Dicionário de Dados:**

| Nome da Variável            | Descrição                                                                                      | Tipo         |
|-----------------------------|------------------------------------------------------------------------------------------------|--------------|
| `Date`                      | Data no formato `dd/mm/aaaa`                                                                  | Categórica   |
| `Time`                      | Hora no formato `hh:mm:ss`                                                                    | Categórica   |
| `Global_active_power`       | Potência ativa média por minuto global doméstica (kW)                                         | Contínua     |
| `Global_reactive_power`     | Potência reativa média por minuto global doméstica (kW)                                       | Contínua     |
| `Voltage`                   | Voltagem média por minuto (V)                                                                 | Contínua     |
| `Global_intensity`          | Intensidade de corrente média por minuto global doméstica (A)                                 | Contínua     |
| `Sub_metering_1`            | Submedição de energia nº 1 (W-h), correspondente à cozinha.                                   | Contínua     |
| `Sub_metering_2`            | Submedição de energia nº 2 (W-h), correspondente à lavanderia.                                | Contínua     |
| `Sub_metering_3`            | Submedição de energia nº 3 (W-h), correspondente a aquecedores e ar-condicionado.             | Contínua     |

---

### Modelagem
- **Técnica de Clusterização:**  
  Utilizamos o algoritmo **K-Means** para segmentar os consumidores em clusters com características semelhantes.
  
- **Pré-processamento:**  
  - Transformação dos tipos de dados para formatos adequados.
  - Normalização com **MinMaxScaler** para evitar problemas relacionados a escalas diferentes.
  
- **Redução de Dimensionalidade:**  
  Aplicação de **PCA** para reduzir o número de variáveis, otimizando a performance do modelo.

- **Escolha do Número de Clusters:**  
  - Análise do **Silhouette Score**:  
    - K=8: 0.3670  
    - K=4: 0.4494 (melhor resultado)
  - Curva de **Elbow** para validar a escolha do número ideal de clusters.

---

### Visualizações e Análises
- **Distribuições Univariadas:** Histogramas e boxplots para entender a distribuição das variáveis numéricas.  
- **Correlação:** Mapa de calor e scatterplots para identificar relações entre variáveis.  
- **Clusters em 3D:** Gráficos tridimensionais para visualizar os grupos formados pelo K-Means.  

---

### Resultados
- **Clusters Identificados:**  
  O número ideal de clusters foi **4**, com perfis distintos de consumo de energia.  
- **Silhouette Score:**  
  O melhor score obtido foi **0.4494**, indicando boa separação entre os grupos.  
- **Interpretação:**  
  Cada cluster representa um perfil de consumo (e.g., baixo consumo em todas as zonas, alto consumo em uma zona específica).

---

## Implantação
Criamos um aplicativo interativo utilizando **Streamlit**.  
- [Link do app](https://clustereletricalgorithm.streamlit.app/)

### Funcionalidades do App:
1. **Upload de Arquivo:** O usuário pode carregar seu próprio dataset.  
2. **Análises Automatizadas:** O app realiza todas as etapas do projeto automaticamente.  
3. **Visualizações Dinâmicas:** Gráficos e tabelas interativos para explorar os insights de consumo.  

---

## Próximos Passos
1. **Validação do Modelo:** Testar os clusters com novos dados.  
2. **Análise de Impacto no Negócio:** Utilizar os clusters para embasar decisões estratégicas.  
3. **Melhorias no App:** Expandir funcionalidades, como upload de novos modelos ou análise comparativa.

---

## Requisitos
- **Linguagem:** Python  
- **Principais Bibliotecas:**  
  - `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`, `scikit-learn`, `streamlit`  

---

