import numpy as np
import pandas as pd
import networkx as nx
import MST as mst
import matriz_euclidiana as me

def gerar_amostras_bootstrap(severity_label, k=1500, T=30):
    """
    Gera k amostras de pacientes, garantindo pelo menos 1 paciente de cada classe.
    Args:
        severity_label: Array ou Series com os labels de severidade (0, 1, 2).
        k: Número de amostras (trajetórias) a gerar.
        T: Tamanho de cada amostra (número de pacientes).
    Returns:
        amostras: Lista de arrays, onde cada array contém 30 índices ORIGINAIS (inteiros) do dataframe.
    """
    
    # 1. Mapear onde estão os pacientes de cada classe
    # Para escabilidade 
    classes_unicas = np.unique(severity_label) # [0, 1, 2]
    # Criar um dicionário: 
    # {0: [índices dos pacientes 0], 1: [índices dos pacientes 1], 2: [índices dos pacientes 2]}
    indices_por_classe = {c: np.where(severity_label == c)[0] for c in classes_unicas}
    
    # Lista com todos os índices possíveis
    todos_indices = np.arange(len(severity_label))
    
    amostras = []
    
    print(f"Iniciando Bootstrap ({k} iterações)...")
    print(f"Classes encontradas para estratificação: {classes_unicas}")
    
    for i in range(k):
        indices_selecionados = []
        
        # Garantir Representatividade (Estratificação) ---
        # Pega aleatoriamente 1 paciente de cada classe (0, 1 e 2)
        for c in classes_unicas:
            idx = np.random.choice(indices_por_classe[c])
            indices_selecionados.append(idx)
            
        # Preencher o resto da amostra ---
        # Se temos 3 classes, já pegamos 3, faltam 27.
        qtde_faltante = T - len(indices_selecionados)
        
        # Pegamos todos os pacientes MENOS os que já escolhemos no passo A
        # (Para evitar ter o mesmo paciente duplicado na mesma amostra)
        # Se o índice ESTIVER na lista indices_selecionados, marque como False. 
        # Se NÃO ESTIVER, marque como True (invert=True)
        mask_disponiveis = np.isin(todos_indices, indices_selecionados, invert=True)
        candidatos = todos_indices[mask_disponiveis] # removeu os que ja escolhemos
        
        # Sorteia o restante (sem reposição)
        resto = np.random.choice(candidatos, qtde_faltante, replace=False)
        
        # Junta as duas partes da amostra: obrigatórios + sorteados 
        amostra_final = np.concatenate([indices_selecionados, resto])
        
        # Embaralhar para que os obrigatórios não fiquem sempre nas primeiras posições
        np.random.shuffle(amostra_final)
        
        amostras.append(amostra_final)
        
    print(f"Concluído. {len(amostras)} amostras geradas.")
    return amostras



def processar_todas_trajetorias(df_norm, severity_label_serie, amostras_indices, qtd_plots=5):
    """
    Itera sobre as amostras, calcula matriz, gera MST e ordena.
    Args:
        df_norm: DataFrame normalizado com os dados dos pacientes.
        severity_label_serie: Series com os labels de severidade (0, 1, 2).
        amostras_indices: Lista de arrays, onde cada array contém índices ORIGINAIS (inteiros) do dataframe.
    Returns:
        trajetorias_finais: Lista de trajetórias, onde cada trajetória é uma lista de índices ordenados pelo pseudo-tempo.
    """
    trajetorias_ordenadas = []
    
    print(f"Iniciando processamento de {len(amostras_indices)} amostras...")

    # LOOP PRINCIPAL
    for i, indices_amostra_atual in enumerate(amostras_indices):
        
        # Pega os pacientes completos que correspondem aos índices(id) da amostra atual 
        df_recorte = df_norm.iloc[indices_amostra_atual]
        
        # Calcula a matriz de distância para essa amostra
        _, df_matriz = me.compute_distance_matrix(df_recorte, sample_size=None)
        
        # 3. MST (usando Prim) 
        grafo_mst = mst.mst(df_matriz) 
        
        # 4. Achar a Raiz
        # Cruza o ID do recorte com a severidade para obter a severidade correta dos pacientes da amostra(recorte)
        id_severity_recorte = severity_label_serie.loc[df_recorte.index]
        
        # Pega quem é 0 (Saudável)(lista dos índices [200, 451, 741, 1002,...])
        candidatos_raiz = id_severity_recorte[id_severity_recorte == 0].index.tolist()
        
        # Sorteia o id da Raiz
        idx_raiz = np.random.choice(candidatos_raiz)
        if i < qtd_plots:
            mst.plotar_mst_amostra(grafo_mst, id_severity_recorte, i)

        # --- ETAPA 4: ORDENAÇÃO (PSEUDO-TEMPO) ---
        candidatos_raiz = id_severity_recorte[id_severity_recorte == 0].index.tolist()
        
        # Garantia de segurança
        if not candidatos_raiz:
            print(f"Aviso Crítico: Amostra {i} sem saudáveis. Pulando.")
            continue 
            
        idx_raiz = np.random.choice(candidatos_raiz)
        
        try:
            distancias = nx.shortest_path_length(grafo_mst, source=idx_raiz, weight='weight')
            ordem_classificada = sorted(distancias.items(), key=lambda x: x[1])
            ids_ordenados = [paciente_id for paciente_id, dist in ordem_classificada]
            trajetorias_ordenadas.append(ids_ordenados)
            
        except nx.NetworkXNoPath:
            print(f"Erro: Grafo desconexo na amostra {i}")

        if (i + 1) % 100 == 0:
            print(f"Progresso: {i + 1}/{len(amostras_indices)}")

        # 5. Ordenar pela MST (pseudo-tempo)

    print("Processamento Finalizado.")
    return trajetorias_ordenadas