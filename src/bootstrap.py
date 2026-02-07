import numpy as np

def gerar_amostras_bootstrap(severity_label, k=1500, T=30):
    """
    Gera amostras (T=30) com Alta Variabilidade nos Graves.
    Regras de Seleção (Cotas Mínimas):
    - Grave (2):    1 a 10 pacientes. (Extrema variação: de caso único a cluster denso)
    - Moderado (1): 5 a 10 pacientes. (Mantém a ponte estável)
    - Saudável (0): 1 a 4 pacientes.  (Raiz mínima)
    
    O restante (T - soma) é preenchido pelo pool global (majoritariamente saudáveis).
    """
    
    classes_unicas = np.unique(severity_label)
    pool_indices = {c: np.where(severity_label == c)[0] for c in classes_unicas}
    all_indices = np.arange(len(severity_label))
    
    amostras = []
    
    print(f"Iniciando Bootstrap | {k} iterações...")
    
    for _ in range(k):
        indices_selecionados = []
        indices_usados_set = set() # p/ nao repetir pacientes na mesma amostra
        
        # Definir as Regras (Cotas) da iteracao atual
        # Criamos uma lista de tuplas: (CLASSE, QUANTIDADE)
        # random.randint(low,high) - high e exclusivo
        configuracao_atual = [
            (2, np.random.randint(1, 11)), # Grave: 1 a 10
            (1, np.random.randint(5, 11)), # Moderado: 5 a 10
            (0, np.random.randint(1, 5))   # Saudável: 1 a 4
        ]
        
        # Loop de Seleção Obrigatória
        for classe, qtd_alvo in configuracao_atual:
            pool = pool_indices[classe] # lista dos indices por classe
            # sorteia a qtd_alvo de pacientes da classe
            selecionados = np.random.choice(pool, size=qtd_alvo, replace=False) 

            indices_selecionados.extend(selecionados)
            indices_usados_set.update(selecionados)
        
        # Preenchimento Natural 
        falta = T - len(indices_selecionados) # (30 - selecionados) 
        
        if falta > 0:
            # Identifica quem sobrou (True para quem NÃO está no set de usados)
            mask_livres = np.isin(all_indices, list(indices_usados_set), invert=True)
            pool_restante = all_indices[mask_livres]

            # Sorteia a quantidade que falta e acrescenta na lista 
            extras = np.random.choice(pool_restante, size=falta, replace=False)
            indices_selecionados.extend(extras)
        
        # Finalização
        amostra_final = np.array(indices_selecionados) # transforma a lista em um array np
        np.random.shuffle(amostra_final) # remover viés de ordem
        amostras.append(amostra_final) # inserir amostra atual na lista de amostras

    return amostras


