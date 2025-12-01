import pandas as pd
import networkx as nx
import numpy as np

def generate_longitudinal_dataset(mst_matrix, root_node_idx, df_real, labels):
    print("1. Convertendo MST para grafo NetworkX (para facilitar navegação)...")
    # A matriz MST do scipy é ótima para cálculo, mas networkx é melhor para achar caminhos
    G = nx.from_scipy_sparse_array(mst_matrix)
    
    print("2. Identificando os nós terminais (Leaf Nodes)...")
    # "Folhas" são nós que têm apenas 1 conexão (fim da linha) e não são a raiz
    # Focamos em folhas que são DOENTES (F, E, ou G avançado) para pegar trajetórias completas
    degrees = dict(G.degree())
    leaf_nodes = [node for node, degree in degrees.items() if degree == 1 and node != root_node_idx]
    
    print(f"Total de trajetórias potenciais encontradas: {len(leaf_nodes)}")
    
    longitudinal_data = []
    
    # Mapeamento reverso para saber quem é quem
    idx_to_class = labels.to_dict()
    
    print("3. Reconstruindo históricos médicos (Backtracking)...")
    
    patient_id_counter = 0
    
    for leaf in leaf_nodes:
        # Só queremos trajetórias que terminam em DOENÇA (F, E, ou H). 
        # Se a trajetória termina em "Saudável", ela é curta e pouco útil para treino.
        final_class = idx_to_class.get(leaf)
        
        # Filtrar: Só gera dados se o paciente final for Doente (2, 3, 4) ou Compensado (1)
        if final_class in [0]: # Pula se terminar em saudável
            continue
            
        try:
            # Acha o caminho único da Raiz até essa Folha
            path = nx.shortest_path(G, source=root_node_idx, target=leaf)
            
            # Se o caminho for muito curto (ex: < 5 passos), talvez seja ruído
            if len(path) < 5:
                continue
                
            # Agora transformamos esse caminho em linhas de um dataset
            for time_step, node_idx in enumerate(path):
                # Pega os dados reais desse paciente-nó
                patient_data = df_real.iloc[node_idx].to_dict()
                
                # Adiciona metadados da simulação
                patient_data['sim_patient_id'] = f"P_{patient_id_counter}" # ID único do paciente virtual
                patient_data['time_step'] = time_step # 0, 1, 2, 3... (Evolução)
                patient_data['real_class'] = idx_to_class.get(node_idx) # A classe real daquele nó
                
                longitudinal_data.append(patient_data)
            
            patient_id_counter += 1
            
        except nx.NetworkXNoPath:
            continue

    # Cria o DataFrame final
    df_longitudinal = pd.DataFrame(longitudinal_data)
    
    print(f"Dataset Longitudinal Gerado com Sucesso!")
    print(f"Pacientes Virtuais Criados: {patient_id_counter}")
    print(f"Total de registros (linhas de tempo): {len(df_longitudinal)}")
    
    return df_longitudinal

# --- EXECUÇÃO ---
# Gera o dataset final
# df_simulated = generate_longitudinal_dataset(mst_matrix, root_node, df_real, labels)

# # Visualizar o "Histórico" de um paciente virtual
# print("\nExemplo de histórico do primeiro paciente virtual gerado:")
# first_patient = df_simulated['sim_patient_id'].iloc[0]
# print(df_simulated[df_simulated['sim_patient_id'] == first_patient][['time_step', 'TSH', 'real_class', 'age']])