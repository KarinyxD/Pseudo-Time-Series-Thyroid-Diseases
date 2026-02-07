import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

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


# --- 2. FUNÇÃO PARA PLOTAR A MATRIZ COM NÚMEROS ---
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


