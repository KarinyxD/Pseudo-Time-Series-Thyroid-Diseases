import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

RANDOM_SEED = 6 #trocar o random_state para variar a amostra

# --- 1. FUNÇÃO PARA CALCULAR A MATRIZ ---
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
    
    print(f"Matriz calculada com sucesso. Dimensões: {dist_matrix.shape}")
    return dist_matrix, df_used


# --- 2. FUNÇÃO PARA PLOTAR A MATRIZ (mapa de calor) ---
def plot_matrix(dist_matrix):
    """
    Plota o heatmap da matriz fornecida.
    Para visualizar a densidade e outliers.
    """
    plt.figure(figsize=(10, 10))
    
    # square=True garante proporção 1:1 dos eixos
    sns.heatmap(dist_matrix, cmap='viridis_r', square=True, 
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": .8})
    
    plt.title(f"Matriz de Distância ({dist_matrix.shape[0]} Pacientes)\n"
              "Cores: Claro = Similar | Escuro = Diferente")
    plt.show()


# --- 3. FUNÇÃO PARA PLOTAR A MATRIZ NUMÉRICA ---
def inspect_numerical_matrix(df_normalized, label, n_samples):
    """
    Gera uma visualização focada nos NÚMEROS da matriz de distância.
    Usa amostragem fixa (RANDOM_SEED).
    """
    
    # 1. Preparação
    df_temp = df_normalized.copy()
    df_temp['label_severity'] = label
    
    # 2. Amostragem com Random State fixo
    df_sample = df_temp.sample(n=n_samples, random_state=RANDOM_SEED)

    # Separar dados e label para processamento
    label_sample = df_sample['label_severity'] # coluna de doenças para legenda
    indices_originais = df_sample.index # id's dos pacientes
    data_sample = df_sample.drop(columns=['label_severity']) 
    
    # 3. CHAMADA DA FUNÇÃO DE CÁLCULO
    # retorna (matriz, df_usado), pegamos só a matriz
    dist_matrix, _ = compute_distance_matrix(data_sample)
    
    # 4. Criar DataFrame Visual (para plotagem)
    nomes_colunas = [f"Pct_{idx} (G{grau})" for idx, grau in zip(indices_originais, label_sample)]
    
    df_matrix_numerica = pd.DataFrame(
        dist_matrix, 
        index=nomes_colunas, 
        columns=nomes_colunas
    )

    # 5. Plotagem
    plt.figure(figsize=(20, 18)) 
    
    sns.heatmap(
        df_matrix_numerica, 
        cmap='viridis_r', 
        annot=True,          # Escreve os números
        fmt=".1f",           # 1 casa decimal
        annot_kws={"size": 8}, 
        square=True,
        cbar_kws={"shrink": .5}
    )
    
    plt.title(f"Matriz Numérica - ({n_samples} Pacientes)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.show()