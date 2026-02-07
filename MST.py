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

def plotar_mst_amostra(grafo_mst, id_severity_recorte, numero_amostra):
    """
    Função auxiliar para visualizar a MST gerada.
    Usa o layout Kamada-Kawai para respeitar as distâncias reais (pesos).
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Layout "Físico" (Respeita o peso das arestas para definir a distância visual)
    pos = nx.kamada_kawai_layout(grafo_mst, weight='weight')
    
    # 2. Mapeamento de Cores
    # 0: Saudável (Verde), 1: Moderado (Laranja), 2: Grave (Vermelho)
    mapa_cores = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    
    # Cria a lista de cores na mesma ordem dos nós do grafo
    lista_cores = []
    for node in grafo_mst.nodes():
        classe = id_severity_recorte.loc[node] # Busca a classe desse ID específico
        lista_cores.append(mapa_cores[classe])

    # 3. Desenho
    nx.draw_networkx_edges(grafo_mst, pos, alpha=0.4, width=1.0)
    nx.draw_networkx_nodes(grafo_mst, pos, node_color=lista_cores, node_size=400, edgecolors='black')
    
    # Adiciona os IDs (opcional, se ficar muito poluído pode comentar)
    nx.draw_networkx_labels(grafo_mst, pos, font_size=8, font_color='black', font_weight='bold')
    
    # 4. Legenda Personalizada
    legenda_elementos = [
        Line2D([0], [0], marker='o', color='w', label='Saudável (0)', markerfacecolor='#2ecc71', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Moderado (1)', markerfacecolor='#f39c12', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Grave (2)', markerfacecolor='#e74c3c', markersize=10)
    ]
    plt.legend(handles=legenda_elementos, loc='best')
    
    plt.title(f"Visualização da MST - Amostra #{numero_amostra}\n(Layout baseado na Similaridade Biológica)")
    plt.axis('off')
    plt.show()


def plotar_trajetoria_mst(grafo_mst, indices_ordenados, num_amostra):
    """
    Gera o gráfico da MST com gradiente de cores baseado na ordem da trajetória.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Definir a posição dos nós (Layout)
    pos = nx.kamada_kawai_layout(grafo_mst) 
    
    # 2. Criar o Mapa de Cores baseado na POSIÇÃO na lista ordenada
    mapa_ordem = {id_pac: i for i, id_pac in enumerate(indices_ordenados)}
    
    # Garante que a cor de cada nó do grafo siga a ordem da trajetória
    cores_nos = [mapa_ordem[no] for no in grafo_mst.nodes()]
    
    # 3. Desenhar a MST
    nx.draw(
        grafo_mst, 
        pos, 
        node_color=cores_nos, 
        cmap=plt.cm.coolwarm, # Azul (Início) -> Vermelho (Fim)
        with_labels=True, 
        node_size=500,
        font_size=8,
        font_color='white'
    )
    
    # Título e Barra de Cores
    plt.title(f"Amostra {num_amostra}: Trajetória Ordenada (Azul=Saudável -> Vermelho=Grave)")
    
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm, 
        norm=plt.Normalize(vmin=0, vmax=len(indices_ordenados))
    )
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Ordem na Trajetória (Pseudotempo)")
    
    plt.show()


def plotar_evolucao_clinica_individual(df_recorte, indices_ordenados, id_severity_recorte):
    # --- 1. Reorganizar os dados na ordem da trajetória ---
    df_ordenado = df_recorte.loc[indices_ordenados].copy()
    severidade_ordenada = id_severity_recorte.loc[indices_ordenados]

    # --- 2. Configurar Cores ---
    mapa_cores = {0: 'blue', 1: 'orange', 2: 'red'}
    cores_pontos = [mapa_cores[s] for s in severidade_ordenada]

    # --- NOMES DAS COLUNAS (AJUSTE AQUI SE DER ERRO) ---
    coluna_tsh = 'TSH'
    coluna_t4 = 'TT4'  
    coluna_t3 = 'T3'  

    # --- 3. Plotar (Agora com 3 linhas) ---
    # Aumentei o figsize para (12, 15) para caber os 3 gráficos bem
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # --- GRÁFICO 1: TSH ---
    ax1.scatter(range(len(df_ordenado)), df_ordenado[coluna_tsh], c=cores_pontos, alpha=0.6, s=20, label='Pacientes')
    ax1.plot(df_ordenado[coluna_tsh].rolling(window=20, center=True).mean().values, color='black', linewidth=2, label='Tendência')
    
    ax1.set_ylabel(f'Nível de {coluna_tsh}')
    ax1.set_title(f'Evolução do {coluna_tsh} (Pseudotempo)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- GRÁFICO 2: T4 ---
    ax2.scatter(range(len(df_ordenado)), df_ordenado[coluna_t4], c=cores_pontos, alpha=0.6, s=20)
    ax2.plot(df_ordenado[coluna_t4].rolling(window=20, center=True).mean().values, color='black', linewidth=2)
    
    ax2.set_ylabel(f'Nível de {coluna_t4}')
    ax2.set_title(f'Evolução do {coluna_t4}')
    ax2.grid(True, alpha=0.3)

    # --- GRÁFICO 3: T3 (O NOVO) ---
    # Verifica se a coluna existe para evitar erro, ou plota mesmo assim se tiver certeza
    if coluna_t3 in df_ordenado.columns:
        ax3.scatter(range(len(df_ordenado)), df_ordenado[coluna_t3], c=cores_pontos, alpha=0.6, s=20)
        ax3.plot(df_ordenado[coluna_t3].rolling(window=20, center=True).mean().values, color='black', linewidth=2)
        ax3.set_ylabel(f'Nível de {coluna_t3}')
        ax3.set_title(f'Evolução do {coluna_t3}')
    else:
        ax3.text(0.5, 0.5, f"Coluna '{coluna_t3}' não encontrada", ha='center')
    
    ax3.grid(True, alpha=0.3)
    
    # O Label do eixo X fica só no último gráfico (o de baixo)
    ax3.set_xlabel('Progresso da Doença (Ordem dos Pacientes na Trajetória)')

    plt.tight_layout()
    plt.show()



def plotar_evolucao_clinica(df_amostra, indices_ordenados, serie_severidade, num_amostra):
    """
    Plota o painel de 3 gráficos (TSH, T4, T3) baseados na ordem da trajetória.
    """
    # 1. Reorganizar os dados na ordem da trajetória
    df_ord = df_amostra.loc[indices_ordenados].copy()
    sev_ord = serie_severidade.loc[indices_ordenados]

    # 2. Configurar Cores e Nomes
    mapa_cores = {0: 'blue', 1: 'orange', 2: 'red'}
    cores_pontos = [mapa_cores[s] for s in sev_ord]
    
    colunas = {
        'tsh': 'TSH',
        't4': 'TT4',
        't3': 'T3'
    }

    # 3. Criar a Figura
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Lista de eixos e colunas para iterar e evitar repetição de código
    config_plots = [
        (ax1, colunas['tsh'], "Evolução do TSH"),
        (ax2, colunas['t4'],  "Evolução do T4"),
        (ax3, colunas['t3'],  "Evolução do T3")
    ]

    for ax, col, titulo in config_plots:
        if col in df_ord.columns:
            # Pontos dos pacientes
            ax.scatter(range(len(df_ord)), df_ord[col], c=cores_pontos, alpha=0.6, s=20)
            
            # Linha de tendência (Média móvel)
            tendencia = df_ord[col].rolling(window=20, center=True).mean()
            ax.plot(range(len(df_ord)), tendencia.values, color='black', linewidth=2)
            
            ax.set_ylabel(f'Nível de {col}')
            ax.set_title(f'{titulo} - Amostra {num_amostra}')
        else:
            ax.text(0.5, 0.5, f"Coluna '{col}' não encontrada", ha='center', va='center')
        
        ax.grid(True, alpha=0.3)

    ax3.set_xlabel('Progresso da Doença (Ordem dos Pacientes na Trajetória)')
    
    plt.tight_layout()
    plt.show()