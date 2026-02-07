import numpy as np

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


