import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

RANDOM_SEED = 61  #trocar o random_state para variar a amostra
# (visualizar a MST): 22, 25, 33, 61


# --- 1. FUNÇÃO AUX PARA CALCULAR A MATRIZ ---
def compute_distance_matrix(df_normalized, sample_size=None):
    """
    Calcula a matriz de distância euclidiana.
    Args:
        df_normalized: DataFrame com dados normalizados.
        sample_size: (Opcional) Inteiro. Se definido, faz uma amostragem aleatória 
    Returns:
        dist_matrix: Matriz quadrada (numpy array) com as distâncias.
        df_used: amostra do DataSet que foi usado para gerar a matriz.
    """
    # Amostragem 
    if sample_size and len(df_normalized) > sample_size:
        print(f"Amostrando {sample_size} pacientes para cálculo da matriz...")
        df_used = df_normalized.sample(n=sample_size, random_state=RANDOM_SEED) #trocar o random_state para variar a amostra
    else:
        df_used = df_normalized.copy()

    # Cálculo da Distância Euclidiana
    dist_vector = pdist(df_used.values, metric='euclidean') #vetor de distancias
    dist_matrix = squareform(dist_vector) # converte para matriz quadrada
    
    # Gera o DataFrame Pandas com os nomes
    df_matrix_formatada = pd.DataFrame(
        dist_matrix,
        index=df_used.index,   # Garante que saberemos quem é o paciente certo
        columns=df_used.index
    )

    return dist_matrix, df_matrix_formatada


# --- 2. FUNÇÃO PARA INSPECIONAR A MATRIZ CALCULADA, GERA DATAFRAME --- 
# escolhe amostragem aleatoriamente
def prepare_distance_dataframe(df_normalized, label, n_samples=None):
    """
    1. Faz a amostragem garantindo que Dados e o Label_Severity fiquem alinhados.
    2. Chama compute_distance_matrix para fazer o cálculo da matriz de distancia euclidiana.
    3. Retorna um DataFrame Pandas com index e colunas nomeados (ex: 'Pct_10 (G0)').
    Args:
        df_normalized: DataFrame dos dados normalizados.
        label: é a coluna label_severity do dataset.
        n_samples: quantidade de pacientes a serem analisados (será a dimensão da matriz).
    """
    # --- PREPARAÇÃO E AMOSTRAGEM ---
    # Unir dados e label severity temporariamente para garantir que, ao sortear,
    # o rótulo do paciente não se perca do exame dele.
    df_temp = df_normalized.copy()
    df_temp['label_severity'] = label
    
    # Se n_samples for definido e menor que o total, fazemos o sorteio
    if n_samples and n_samples < len(df_temp):
        df_sample = df_temp.sample(n=n_samples, random_state=RANDOM_SEED)
    else:
        df_sample = df_temp

    # Separamos de volta: Dados para um lado, Label para o outro
    label_sample = df_sample['label_severity'] 
    indices_originais = df_sample.index 
    data_sample = df_sample.drop(columns=['label_severity']) 
    
    # --- CHAMADA DA compute_distance_matriz ---
    # Sample_size=None porque a amostragem já foi feita
    # Retorna a matriz numpy (sem rótulos)
    dist_matrix_numpy, _ = compute_distance_matrix(data_sample, sample_size=None)
    
    # --- FORMATAÇÃO DO DATAFRAME FINAL ---
    # Cria a lista de nomes: "Pct_ID (G_GRAU)"
    nomes_colunas = [f"Pct_{idx} (G{grau})" for idx, grau in zip(indices_originais, label_sample)]
    
    # Gera o DataFrame Pandas com os nomes
    df_matrix_formatada = pd.DataFrame(
        dist_matrix_numpy, 
        index=nomes_colunas, 
        columns=nomes_colunas
    )
    
    return df_matrix_formatada


# --- 3. FUNÇÃO PARA INSPECIONAR A MATRIZ NUMÉRICA, GERA DATAFRAME --- 
# escolhe amostragem incluindo quantidades especificas de cada grau da doenca
def prepare_distance_dataframe_amost(df_normalized,label,n_g0=45,n_g1=20,n_g2=20):
    """
    1. Faz a amostragem selecionando a quantidade especificada nos parametros, de cada grau de severidade.
    2. Garante que os Dados e o Label_Severity fiquem alinhados.
    2. Chama compute_distance_matrix para fazer o cálculo da matriz de distancia euclidiana.
    3. Retorna um DataFrame Pandas com index e colunas nomeados (ex: 'Pct_10 (G0)').
    """
    df_temp = df_normalized.copy()
    df_temp['label_severity'] = label

    df_g0 = df_temp[df_temp['label_severity'] == 0]
    df_g1 = df_temp[df_temp['label_severity'] == 1]
    df_g2 = df_temp[df_temp['label_severity'] == 2]
    df_g3 = df_temp[df_temp['label_severity'] == 3]
    df_g4 = df_temp[df_temp['label_severity'] == 4]

    sample_g4 = df_g4                          # todos (8)
    sample_g3 = df_g3                          # todos (1)
    sample_g2 = df_g2.sample(
        n=min(n_g2, len(df_g2)),
        random_state=RANDOM_SEED
    )
    sample_g1 = df_g1.sample(
        n=min(n_g1, len(df_g1)),
        random_state= RANDOM_SEED
    )
    sample_g0 = df_g0.sample(
        n=min(n_g0, len(df_g0)),
        random_state=RANDOM_SEED
    )

    df_sample = pd.concat(
        [sample_g4, sample_g3, sample_g2, sample_g1, sample_g0]
    ).sample(frac=1, random_state=RANDOM_SEED) 

    label_sample = df_sample['label_severity']
    indices_originais = df_sample.index
    data_sample = df_sample.drop(columns=['label_severity'])

    print("--- Amostragem Estratificada ---")
    print(label_sample.value_counts().sort_index())

    # MATRIZ DE DISTÂNCIAS
    # Sample_size=None porque a amostragem já foi feita
    # Retorna a matriz numpy (sem rótulos)
    dist_matrix_numpy, _ = compute_distance_matrix(data_sample, sample_size=None)

    # --- FORMATAÇÃO DO DATAFRAME FINAL ---
    # Cria a lista de nomes: "Pct_ID (G_GRAU)"
    nomes_colunas = [f"Pct_{idx} (G{grau})" for idx, grau in zip(indices_originais, label_sample)]
    
    # Gera o DataFrame Pandas com os nomes
    df_matrix_formatada = pd.DataFrame(
        dist_matrix_numpy, 
        index=nomes_colunas, 
        columns=nomes_colunas
    )

    return df_matrix_formatada


# --- 4. FUNÇÃO PARA PLOTAR MAPA DE CALOR (VISÃO GERAL / SEM NÚMEROS) ---
def plot_overview_heatmap(df_matrix):
    """
    Plota um heatmap SOMENTE VISUAL (sem números e sem nomes nos eixos).
    Args:
        df_matrix: DataFrame da matriz de distâncias.
    """
    plt.figure(figsize=(12, 10))
    
    # Heatmap Otimizado
    sns.heatmap(
        df_matrix, 
        cmap='viridis_r', 
        annot=False,          # <--- Não escreve números
        square=True, 
        xticklabels=False,    # Esconde os nomes
        yticklabels=False,    
        cbar_kws={"shrink": .8, "label": "Distância Euclidiana"}
    )
    
    n_pacientes = df_matrix.shape[0]
    
    plt.title(f"Visão Macro: Matriz de Distância ({n_pacientes} Pacientes)\n"
              "Cores: Claro (Amarelo) = Próximo | Escuro (Roxo) = Distante")
    
    plt.tight_layout() 
    plt.show()


# --- 5. FUNÇÃO PARA PLOTAR A MATRIZ COM NÚMEROS ---
def plot_numerical_matrix(df_matrix):
    """
    Recebe o DataFrame já formatado e exibe o Heatmap com números.
    """
    plt.figure(figsize=(20, 18)) 
    
    sns.heatmap(
        df_matrix, 
        cmap='viridis_r', 
        annot=True,          
        fmt=".1f",           
        annot_kws={"size": 8}, 
        square=True,
        cbar_kws={"shrink": .5}
    )
    
    n_samples = len(df_matrix)
    plt.title(f"Matriz Numérica - ({n_samples} Pacientes)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.show()


