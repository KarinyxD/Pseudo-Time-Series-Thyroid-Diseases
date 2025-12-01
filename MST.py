from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

def build_mst_and_root(df_normalized, labels):
    print("1. Calculando Matriz de Distância Euclidiana (pode demorar alguns segundos)...")
    # pdist calcula a distância entre todos os pares de linhas
    dist_array = pdist(df_normalized.values, metric='euclidean')
    dist_matrix = squareform(dist_array)
    
    print("2. Construindo a Minimum Spanning Tree (MST)...")
    # Isso retorna uma matriz esparsa onde só as conexões essenciais existem
    mst_matrix = minimum_spanning_tree(dist_matrix)
    
    print("3. Identificando o Nó Raiz (O paciente 'mais saudável')...")
    # Estratégia: Pegar o centroide da classe '-' (0) e achar o paciente mais próximo dele
    # Isso evita pegar um outlier como raiz.
    
    # Filtra indices onde a classe é '-' (0)
    healthy_indices = np.where(labels == 0)[0] 
    
    # Calcula o paciente médio saudável (centroide)
    healthy_centroid = df_normalized.iloc[healthy_indices].mean().values
    
    # Acha qual paciente real tem a menor distância para esse centroide
    # Usamos cdist para medir a distância de todos os saudáveis até o centroide
    from scipy.spatial.distance import cdist
    distances_to_centroid = cdist(df_normalized.iloc[healthy_indices].values, [healthy_centroid])
    
    # O índice relativo dentro do grupo de saudáveis
    closest_relative_idx = np.argmin(distances_to_centroid)
    
    # O índice real no dataframe original
    root_node_idx = healthy_indices[closest_relative_idx]
    
    print(f"Nó Raiz identificado: Índice {root_node_idx}")
    print(f"Valores do paciente Raiz (Padronizados):\n{df_normalized.iloc[root_node_idx]}")
    
    return mst_matrix, root_node_idx, dist_matrix

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection

def plot_mst_pca(df_normalized, mst_matrix, labels, root_node_idx):
    print("1. Reduzindo dimensionalidade para 2D (PCA) para visualização...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_normalized)
    
    # Extrair coordenadas X e Y
    x = coords[:, 0]
    y = coords[:, 1]
    
    print("2. Preparando as conexões da árvore...")
    # mst_matrix é uma matriz esparsa. .nonzero() nos dá as conexões (quem liga com quem)
    rows, cols = mst_matrix.nonzero()
    
    # Criar uma lista de segmentos de linha [(x1, y1), (x2, y2)]
    # Isso é muito mais rápido que usar plt.plot num loop
    lines = []
    for i, j in zip(rows, cols):
        p1 = (x[i], y[i])
        p2 = (x[j], y[j])
        lines.append([p1, p2])
        
    print("3. Gerando o gráfico...")
    plt.figure(figsize=(14, 10))
    
    # A. Desenhar as linhas da árvore (MST)
    lc = LineCollection(lines, colors='gray', alpha=0.3, linewidths=0.5, zorder=1)
    plt.gca().add_collection(lc)
    
    # B. Desenhar os pacientes (Nós)
    # Cores baseadas na classe
    class_colors = {0: 'tab:blue', 1: 'tab:olive', 2: 'tab:orange', 3: 'tab:red', 4: 'tab:purple'}
    # Mapear labels numéricos de volta para cores
    colors = [class_colors.get(l, 'gray') for l in labels]
    
    scatter = plt.scatter(x, y, c=colors, s=15, alpha=0.8, zorder=2, label='Pacientes')
    
    # C. Destacar a Raiz (Root)
    plt.scatter(x[root_node_idx], y[root_node_idx], c='black', s=200, marker='*', edgecolors='white', zorder=3, label='RAIZ (Início)')

    # Legenda manual para ficar bonito
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='Saudável (-)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:olive', label='Compensado (G)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', label='Primário (F)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='Severo (E)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:purple', label='Secundário (H)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', label='Start Node', markersize=15),
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title('Visualização da Minimum Spanning Tree (Projetada via PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.tight_layout()
    plt.show()

# --- EXECUÇÃO ---
# Precisa ter rodado os passos anteriores para ter essas variáveis
# plot_mst_pca(df_norm, mst_matrix, labels, root_node)

# Rodando a parte 2
# mst_matrix, root_node, full_dist_matrix = build_mst_and_root(df_norm, labels)