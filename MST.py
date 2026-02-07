import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import MDS
from collections import deque

# Cria o grafo e calcula a MST usando o algoritmo de Prim
def mst(df_matrix_numerica):
    """
    Recebe o DataFrame da matriz de distâncias e retorna o Grafo da MST.
    """
    # Transforma o DataFrame em um Grafo Completo (Todos ligados a todos)
    G_completo = nx.from_pandas_adjacency(df_matrix_numerica)
    
    # Calcula a Árvore Geradora Mínima (MST)
    G_mst = nx.minimum_spanning_tree(G_completo, algorithm='prim')
    
    print(f"--- MST Calculada ---")
    print(f"Nós (Pacientes): {G_mst.number_of_nodes()}")
    print(f"Arestas (Conexões): {G_mst.number_of_edges()}")
    
    return G_mst


def plot_mst_graph(G_mst):
    """
    Recebe um objeto Grafo (MST) e plota a visualização.
    """
    plt.figure(figsize=(15, 12))
    
    # Layout (O algoritmo de força que espalha os nós para não ficarem amontoados)
    pos = nx.spring_layout(G_mst, weight='weight')
    
    # Lógica de Cores (Baseada no nome/grau")
    # Definimos uma cor para cada grau
    colors = []
    for node_name in G_mst.nodes():
        if "(G0)" in node_name:
            colors.append('#2ecc71') # Verde (Saudável)
        elif "(G1)" in node_name:
            colors.append('#f1c40f') # Amarelo
        elif "(G2)" in node_name:
            colors.append('#e67e22') # Laranja
        elif "(G3)" in node_name:
            colors.append('#e74c3c') # Vermelho
        elif "(G4)" in node_name:
            colors.append('#8e44ad') # Roxo (Secundário)
        else:
            colors.append('gray')    # Caso não ache o rótulo

    # Desenha os nós 
    nx.draw_networkx_nodes(G_mst, pos, node_size=300, node_color=colors, alpha=0.9, edgecolors='black')
    
    # Desenha as linhas (arestas)
    nx.draw_networkx_edges(G_mst, pos, alpha=0.4, width=1.5, edge_color='gray')
    
    # Desenha os rótulos (Somente se tiver menos de 100 nós)
    if len(G_mst.nodes()) < 100:
        nx.draw_networkx_labels(G_mst, pos, font_size=8, font_weight='bold')

    # Legenda Manual
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='G0: Saudável', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f', label='G1: Leve', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', label='G2: Moderado', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='G3: Severo', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', label='G4: Secundário', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Árvore Geradora Mínima (MST)\nConexões Biológicas Mais Fortes", fontsize=14)
    plt.axis('off')
    plt.show()


def plot_mds_full_mst(G_mst):
    """
    Pega a MST inteira, calcula as distâncias reais (Dijkstra) 
    e plota o mapa MDS.
    """
    print(f"--- Análise Geométrica (Grafo Completo) ---")
    print(f"Total de Pacientes: {G_mst.number_of_nodes()}")
    
    # 1. PREPARAR DADOS
    # Transformamos os nós em uma lista fixa para garantir a ordem
    nodes = list(G_mst.nodes())
    n = len(nodes)
    
    # Se tiver muita gente, avisa (opcional)
    if n > 500:
        print("Aviso: Muitos nós. O cálculo pode demorar um pouco.")

    # 2. CALCULAR MATRIZ DE DISTÂNCIAS (DIJKSTRA)
    # Isso mede a distância andando pelas linhas da árvore
    print("Calculando distâncias entre todos os pares (Dijkstra)...")
    dist_matrix = np.zeros((n, n))
    
    # all_pairs_dijkstra retorna um gerador, convertemos para dict para acesso rápido
    all_distances = dict(nx.all_pairs_dijkstra_path_length(G_mst, weight='weight'))
    
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            # Preenche a matriz: distância do paciente 'u' até o paciente 'v'
            dist_matrix[i, j] = all_distances[u][v]

    # 3. APLICAR O MDS (O Cartógrafo)
    print("Rodando MDS...")
    mds = MDS(
        n_init= 1, # n para evitar ficar preso em minimos locais
        n_components=2, 
        dissimilarity='precomputed', 
        random_state=42, 
        normalized_stress='auto'
    )
    
    coords = mds.fit_transform(dist_matrix)
    
    # Cria o dicionário de posições para o NetworkX desenhar
    pos = {nodes[i]: coords[i] for i in range(n)}

    # 4. DEFINIR CORES (Baseado no nome G0, G1...)
    colors = []
    for node_name in nodes:
        if "(G0)" in node_name: colors.append('#2ecc71')   # Verde
        elif "(G1)" in node_name: colors.append('#f1c40f') # Amarelo
        elif "(G2)" in node_name: colors.append('#e67e22') # Laranja
        elif "(G3)" in node_name: colors.append('#e74c3c') # Vermelho
        elif "(G4)" in node_name: colors.append('#8e44ad') # Roxo
        else: colors.append('gray')

    # 5. DESENHAR
    plt.figure(figsize=(15, 12))
    
    # Desenha os nós
    nx.draw_networkx_nodes(G_mst, pos, node_size=300, node_color=colors, edgecolors='black', alpha=0.9)
    
    # Desenha as arestas (linhas da árvore)
    nx.draw_networkx_edges(G_mst, pos, alpha=0.3, width=1.5, edge_color='gray')
    
    # Rótulos (Só se não estiver muito poluído)
    if n < 150:
        nx.draw_networkx_labels(G_mst, pos, font_size=8, font_weight='bold')

    # Legenda Manual
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='G0: Saudável', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f', label='G1: Leve', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', label='G2: Moderado', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='G3: Severo', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', label='G4: Secundário', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Visualização MDS da Árvore Completa", fontsize=14)
    plt.axis('off')
    plt.show()
