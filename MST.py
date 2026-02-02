import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import MDS
from collections import deque

def compute_mst(df_matrix_numerica):
    """
    Recebe o DataFrame da matriz de distâncias e retorna o Grafo da MST.
    Algoritmo: Prim (ideal para grafos densos onde todos se conectam).
    """
    # 1. Transforma o DataFrame em um Grafo Completo (Todos ligados a todos)
    G_completo = nx.from_pandas_adjacency(df_matrix_numerica)
    
    # 2. Calcula a Árvore Geradora Mínima (MST)
    G_mst = nx.minimum_spanning_tree(G_completo, algorithm='prim')
    
    print(f"--- MST Calculada ---")
    print(f"Nós (Pacientes): {G_mst.number_of_nodes()}")
    print(f"Arestas (Conexões): {G_mst.number_of_edges()}")
    
    return G_mst


def plot_mst_graph(G_mst):
    """
    Recebe um objeto Grafo (MST) e plota a visualização.
    Colore os nós baseado na gravidade (G0 a G4) encontrada no nome do nó.
    """
    plt.figure(figsize=(15, 12))
    
    # 1. Layout (O algoritmo de força que espalha os nós para não ficarem amontoados)
    # k=0.5 controla o espaçamento; seed=42 garante que o desenho não mude se rodar de novo
    pos = nx.spring_layout(G_mst, k=0.5, seed=42)
    
    # 2. Lógica de Cores (Baseada no nome "Pct_123 (G0)")
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

    # 3. Desenhar a Árvore
    # Desenha os nós
    nx.draw_networkx_nodes(G_mst, pos, node_size=300, node_color=colors, alpha=0.9, edgecolors='black')
    
    # Desenha as linhas (arestas)
    nx.draw_networkx_edges(G_mst, pos, alpha=0.4, width=1.5, edge_color='gray')
    
    # Desenha os rótulos (Só desenha se tiver menos de 100 para não poluir)
    if len(G_mst.nodes()) < 100:
        nx.draw_networkx_labels(G_mst, pos, font_size=8, font_weight='bold')

    # Legenda Manual (Para saber quem é quem)
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


def plot_mst_graph_sample_mds(G_mst, sample_size=100, random_state=8):

    """
    Análise geométrica real da MST usando MDS,
    aplicada a uma SUBÁRVORE CONECTADA da MST.

    Distância = menor caminho ponderado
    """
    random.seed(random_state)

    # ============================
    # 1. EXTRAIR SUBÁRVORE CONECTADA (BFS)
    # ============================
    root = random.choice(list(G_mst.nodes()))

    visited = set([root])
    queue = deque([root])

    while queue and len(visited) < sample_size:
        current = queue.popleft()
        for neighbor in G_mst.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= sample_size:
                    break

    sampled_nodes = list(visited)
    G_sub = G_mst.subgraph(sampled_nodes)

    print(f"--- Análise Geométrica (Subárvore Conectada) ---")
    print(f"Nó raiz: {root}")
    print(f"Nós na subárvore: {G_sub.number_of_nodes()}")
    print(f"Arestas na subárvore: {G_sub.number_of_edges()}")

    # ============================
    # 2. MATRIZ DE DISTÂNCIAS REAIS
    # ============================
    nodes = list(G_sub.nodes())
    n = len(nodes)

    dist_matrix = np.zeros((n, n))

    all_distances = dict(
        nx.all_pairs_dijkstra_path_length(G_sub, weight='weight')
    )

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            dist_matrix[i, j] = all_distances[u][v]

    # ============================
    # 3. MDS
    # ============================
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=random_state,
        n_init=1
    )

    coords = mds.fit_transform(dist_matrix)
    pos = {nodes[i]: coords[i] for i in range(n)}

    # ============================
    # 4. CORES DOS NÓS
    # ============================
    colors = []
    for node_name in nodes:
        if "(G0)" in node_name:
            colors.append('#2ecc71')
        elif "(G1)" in node_name:
            colors.append('#f1c40f')
        elif "(G2)" in node_name:
            colors.append('#e67e22')
        elif "(G3)" in node_name:
            colors.append('#e74c3c')
        elif "(G4)" in node_name:
            colors.append('#8e44ad')
        else:
            colors.append('gray')

    # ============================
    # 5. DESENHO
    # ============================
    plt.figure(figsize=(15, 12))

    nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_size=350,
        node_color=colors,
        alpha=0.9,
        edgecolors='black'
    )

    nx.draw_networkx_edges(
        G_sub,
        pos,
        alpha=0.5,
        width=1.5,
        edge_color='gray'
    )

    if n < 100:
        nx.draw_networkx_labels(
            G_sub,
            pos,
            font_size=8,
            font_weight='bold'
        )

    # ============================
    # 6. LEGENDA
    # ============================
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='G0: Saudável', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f', label='G1: Leve', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', label='G2: Moderado', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='G3: Severo', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', label='G4: Secundário', markersize=10)
    ]

    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(
        "Análise Geométrica Real da MST (MDS em Subárvore Conectada)",
        fontsize=14
    )

    plt.axis('off')
    plt.show()
