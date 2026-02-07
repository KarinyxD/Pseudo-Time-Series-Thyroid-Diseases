import numpy as np

def gerar_amostras_bootstrap(severity_label, k=1500, T=30):
    """
    Gera k amostras de pacientes, garantindo pelo menos 1 paciente de cada classe.
    Args:
        severity_label: Array ou Series com os labels de severidade (0, 1, 2).
        k: Número de amostras (trajetórias) a gerar.
        T: Tamanho de cada amostra (número de pacientes).
    Returns:
        amostras: Lista de arrays, onde cada array contém 30 índices ORIGINAIS do dataframe.
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