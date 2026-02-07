from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import MST as mst
import matriz_euclidiana as me

def gerar_amostras_bootstrap(severity_label, k=1500, T=30):
    """
    Gera k amostras de pacientes, garantindo BALANCEAMENTO entre classes.
    Args:
        severity_label: Array ou Series com os labels de severidade (0, 1, 2).
        k: Número de amostras (trajetórias) a gerar.
        T: Tamanho de cada amostra (número de pacientes).
    Returns:
        amostras: Lista de arrays, onde cada array contém índices ORIGINAIS.
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
    
    print(f"Iniciando Bootstrap Estratificado ({k} iterações)...")
    print(f"Classes: {classes_unicas} | Tamanho Amostra: {T}")
    
    # Define quantos pegar de cada classe para ficar balanceado
    # Ex: Se T=30 e temos 3 classes, cota = 10 por classe.
    cota_por_classe = T // len(classes_unicas)
    
    for i in range(k):
        indices_selecionados = []
        
        # Garantir Representatividade (Estratificação) ---
        # AGORA PEGA A COTA INTEIRA (ex: 10) de cada classe, não apenas 1.
        for c in classes_unicas:
            pool_classe = indices_por_classe[c]
            
            # Sorteia 'cota' pacientes dessa classe (ex: 10 saudáveis)
            # replace=False garante que não repetimos o mesmo paciente na mesma amostra
            # (Assumindo que você tem pelo menos 'cota' pacientes no total do dataset para cada classe)
            idx = np.random.choice(pool_classe, size=cota_por_classe, replace=False)
            indices_selecionados.extend(idx)
            
        # Preencher o resto da amostra ---
        # Se T=30 e pegamos 10+10+10, sobra 0. 
        # Se T=32, sobram 2 (que serão preenchidos aleatoriamente).
        qtde_faltante = T - len(indices_selecionados)
        
        if qtde_faltante > 0:
            # Pegamos todos os pacientes MENOS os que já escolhemos no passo anterior
            # (Para evitar ter o mesmo paciente duplicado na mesma amostra)
            # Se o índice ESTIVER na lista indices_selecionados, marque como False. 
            mask_disponiveis = np.isin(todos_indices, indices_selecionados, invert=True)
            candidatos = todos_indices[mask_disponiveis] 
            
            # Sorteia o restante (sem reposição)
            resto = np.random.choice(candidatos, qtde_faltante, replace=False)
            
            # Junta as duas partes da amostra: obrigatórios + sorteados 
            amostra_final = np.concatenate([indices_selecionados, resto])
        else:
            amostra_final = np.array(indices_selecionados)
        
        # Embaralhar para que os obrigatórios não fiquem sempre nas primeiras posições
        np.random.shuffle(amostra_final)
        
        amostras.append(amostra_final)
        
    print(f"Concluído. {len(amostras)} amostras balanceadas geradas.")
    return amostras


def processar_todas_trajetorias(df_norm, severity_label_serie, amostras_indices, qtd_plots=3):
    """
    Itera sobre as amostras, calcula matriz, gera MST e ordena.
    Args:
        df_norm: DataFrame normalizado com os dados dos pacientes.
        severity_label_serie: Series com os labels de severidade (0, 1, 2).
        amostras_indices: Lista de arrays, onde cada array contém índices ORIGINAIS (inteiros) do dataframe.
        qtd_plots: quantidade de amostras (primeiras) que serão plotadas.
    Returns:
        trajetorias_finais: Lista de trajetórias, onde cada trajetória é uma lista de índices ordenados pelo pseudo-tempo.
    """
    trajetorias_finais = []

    print(f"Iniciando processamento de {len(amostras_indices)} amostras...")

    # LOOP PRINCIPAL
    for i, indices_amostra_atual in enumerate(amostras_indices):
        
        # Pega os pacientes completos que correspondem aos índices(id) da amostra atual 
        df_recorte = df_norm.iloc[indices_amostra_atual]
        
        # Calcula a matriz de distância para essa amostra
        _, df_matriz = me.compute_distance_matrix(df_recorte, sample_size=None)
        
        # MST (usando Prim) 
        grafo_mst = mst.mst(df_matriz)
        
        # Cruza o ID do recorte com a severidade para obter a severidade correta dos pacientes da amostra(recorte)
        id_severity_recorte = severity_label_serie.loc[df_recorte.index]
        if i < qtd_plots:
            mst.plotar_mst_amostra(grafo_mst, id_severity_recorte, i)

        # Achar a Raiz - (Medóide Euclidiano) - Centro do Cluster Saudavel
        # Pega quem é 0 (Saudável)(lista dos índices ex:[200, 451, 741, 1002,...])
        candidatos_raiz = id_severity_recorte[id_severity_recorte == 0].index.tolist()
        # Pega os dados dos pacientes saudáveis
        dados_saudaveis = df_recorte.loc[candidatos_raiz]
        # Calcula a (Média) das colunas (centro ideal)
        centro_ideal = dados_saudaveis.mean(axis=0)
        # Calcula a distância de cada saudável real até esse centro ideal
        # (linalg.norm faz o cálculo da distância Euclidiana)
        # lista de "desvios" da média de cada paciente
        distancias_ao_centro = np.linalg.norm(dados_saudaveis - centro_ideal, axis=1) 
        # Encontra a posição do paciente com a MENOR distância (o mais central)
        posicao_melhor_candidato = np.argmin(distancias_ao_centro)
        # Obtém o id do vértice inicial
        idx_raiz = candidatos_raiz[posicao_melhor_candidato]
        
        # Calcular distâncias na árvore (Dijkstra)
        # retorna um dicionário: {id_paciente: distancia, ...}
        dists_dict = nx.shortest_path_length(grafo_mst, source=idx_raiz, weight='weight')
        # Cria uma lista de tuplas para ordenar
        lista_para_ordenar = []
        for paciente_id, distancia_valor in dists_dict.items():
            # Pega a severidade do paciente (0, 1 ou 2)
            severidade_paciente = id_severity_recorte.loc[paciente_id]
            # Adiciona na lista: (Severidade, Distância, ID)
            # A ORDEM DENTRO DA TUPLA É IMPORTANTE
            lista_para_ordenar.append((severidade_paciente, distancia_valor, paciente_id))
            
        # 7. Ordenação 
        # 1º Critério: Severidade (0 vem antes de 1, que vem antes de 2)
        # 2º Critério: Distância (do menor para o maior dentro da mesma severidade)
        lista_para_ordenar.sort()
        
        # Separa apenas os IDs dos pacientes na ordem correta da trajetoria
        indices_ordenados = []
        for item in lista_para_ordenar:
            paciente_id = item[2]  # Pega o ID que está na posicao 2 da tupla
            indices_ordenados.append(paciente_id)

        trajetorias_finais.append(indices_ordenados)

        if i < qtd_plots:
            mst.plotar_trajetoria_mst(grafo_mst, indices_ordenados, i) 
            mst.plotar_evolucao_clinica_individual(df_recorte, indices_ordenados, id_severity_recorte)

        

    print("Processamento Finalizado.")
    return trajetorias_finais


def exportar_trajetorias_com_severidade(df_original, trajetorias_finais, id_severity_map, nome_arquivo="csv/trajetorias.csv"):
    """
    df_original: DataFrame com valores reais (TSH, T4, T3, ...).
    trajetorias_finais: Lista de listas com os IDs ordenados (o retorno da sua função).
    id_severity_map: Series ou dicionário que mapeia ID do paciente -> Severidade (0, 1, 2).
    """
    lista_dfs_rodadas = []

    for i, trajetoria in enumerate(trajetorias_finais):
        # 1. Filtra os dados reais na ordem da trajetória decidida
        df_rodada = df_original.loc[trajetoria].copy()
        
        # 2. Adiciona o ID da Trajetória (seu bootstrap_run)
        df_rodada['id_trajetoria'] = i
        
        # 3. Adiciona a posição (0, 1, 2... N) para saber a sequência
        df_rodada['posicao_trajetoria'] = range(len(trajetoria))
        
        # 4. Adiciona a SEVERIDADE real de cada paciente
        # O .map() vai buscar no seu dicionário/series de severidade o rótulo de cada ID
        df_rodada['severidade'] = df_rodada.index.map(id_severity_map)
        
        # 5. Garante que o ID do paciente não suma (caso ele seja o índice)
        df_rodada['paciente_id'] = df_rodada.index
        
        lista_dfs_rodadas.append(df_rodada)

    # Junta todas as rodadas em um único arquivo
    df_final = pd.concat(lista_dfs_rodadas, ignore_index=True)
    
    # Salva em CSV formatado para Excel Brasil
    df_final.to_csv(nome_arquivo, index=False, sep=';', decimal=',')
    
    print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")
    return df_final