import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# --- 1. FUNÇÃO PARA CALCULAR A MATRIZ ---
def compute_distance_matrix(df_normalized, sample_size=None):
    """
    Calcula a matriz de distância euclidiana para o dataset.
    Baseado na metodologia de Puccio et al. para criar a base da Pseudo Time Series.
    
    Args:
        df_normalized: DataFrame com dados normalizados.
        sample_size: (Opcional) Inteiro. Se definido, faz uma amostragem aleatória 
                     antes de calcular (útil para datasets > 5000 linhas).
    
    Returns:
        dist_matrix: Matriz quadrada (numpy array) com as distâncias.
        df_used: O DataFrame (ou amostra) que foi usado para gerar a matriz.
                 (Importante retornar isso para sabermos quem são os pacientes da matriz)
    """
    # Amostragem (se necessário)
    if sample_size and len(df_normalized) > sample_size:
        print(f"Amostrando {sample_size} pacientes para cálculo da matriz...")
        df_used = df_normalized.sample(n=sample_size, random_state=42)
    else:
        df_used = df_normalized.copy()

    # Cálculo da Distância Euclidiana
    # O artigo especifica o uso de distâncias Euclidianas para a matriz completa 
    dist_vector = pdist(df_used.values, metric='euclidean')
    dist_matrix = squareform(dist_vector)
    
    print(f"Matriz calculada com sucesso. Dimensões: {dist_matrix.shape}")
    return dist_matrix, df_used


# --- 2. FUNÇÃO PARA PLOTAR A MATRIZ GERAL (Heatmap Completo) ---
def plot_full_matrix(dist_matrix):
    """
    Plota o heatmap da matriz inteira fornecida.
    Útil para visualizar a densidade global e outliers.
    """
    plt.figure(figsize=(10, 10))
    
    # 'square=True' garante proporção 1:1 (simetria visual)
    sns.heatmap(dist_matrix, cmap='viridis_r', square=True, 
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": .8})
    
    plt.title(f"Matriz de Distância Global ({dist_matrix.shape[0]} Pacientes)\n"
              "Cores: Claro = Similar | Escuro = Diferente")
    plt.show()


# --- 3. FUNÇÃO PARA PLOTAR AMOSTRA  (45 Pacientes) ---
def plot_ordered_sample_matrix(df_normalized, labels, n_samples=45):
    """
    Seleciona uma amostra pequena, ORDENA por gravidade e plota a matriz.
    Isso valida visualmente se os grupos (saudável vs doente) estão se formando.
    
    Args:
        df_normalized: O dataframe completo normalizado.
        labels: A série com os labels de severidade (0 a 4).
        n_samples: Tamanho da amostra para visualização (Padrão: 45).
    """
    # 1. Unir dados e labels temporariamente
    df_temp = df_normalized.copy()
    df_temp['label_severity'] = labels
    
    # 2. Amostragem Estratificada
    # Tenta pegar um pouco de cada classe para o gráfico ficar rico
    try:
        # Pega amostra proporcional de cada grupo de severidade
        df_sample = pd.concat([
            group.sample(n=min(len(group), max(1, n_samples // 5)), random_state=42)
            for _, group in df_temp.groupby('label_severity')
        ])
        # Se a amostragem estratificada resultou em menos que o desejado, completa aleatoriamente
        if len(df_sample) < n_samples:
             restante = n_samples - len(df_sample)
             # Evita duplicatas pegando do que sobrou
             pool = df_temp.drop(df_sample.index)
             if not pool.empty:
                 extra = pool.sample(n=min(len(pool), restante), random_state=42)
                 df_sample = pd.concat([df_sample, extra])
                 
    except Exception as e:
        print(f"Aviso: Amostragem estratificada falhou ({e}), usando aleatória simples.")
        df_sample = df_temp.sample(n=n_samples, random_state=42)

    # 3. ORDENAÇÃO (Crucial para ver os blocos)
    # Ordena do Saudável (0) -> Doente (4) 
    df_sample = df_sample.sort_values(by='label_severity')
    
    # Separar novamente para cálculo
    labels_sample = df_sample['label_severity']
    data_sample = df_sample.drop(columns=['label_severity'])
    
    # 4. Calcular matriz dessa amostra pequena
    matrix_sample = squareform(pdist(data_sample, metric='euclidean'))
    
    # 5. Plotar
    plt.figure(figsize=(8, 8))
    sns.heatmap(matrix_sample, cmap='viridis_r', square=True,
                xticklabels=False, yticklabels=False, cbar=True)
    
    plt.title(f"Matriz Ordenada por Severidade (Amostra de {len(df_sample)} Pacientes)\n"
              "Eixo: Saudável (0) --> Avançado (4)")
    plt.xlabel("Pacientes (Ordenados)")
    plt.ylabel("Pacientes (Ordenados)")
    plt.show()
    
    # Retorna os dados usados caso queira usar na MST depois
    return matrix_sample, labels_sample

def inspect_numerical_matrix(df_normalized, labels, n_samples=45):
    """
    Gera uma visualização focada nos NÚMEROS da matriz de distância.
    
    Retorna:
        df_matrix: Um DataFrame do Pandas com os valores numéricos exatos,
                   ordenados por gravidade.
    """
    
    # 1. Preparação (Igual à anterior: Amostra e Ordena)
    df_temp = df_normalized.copy()
    df_temp['label_severity'] = labels
    
    # Tenta amostragem estratificada, senão aleatória
    try:
        df_sample = pd.concat([
            group.sample(n=min(len(group), max(1, n_samples // 5)), random_state=42)
            for _, group in df_temp.groupby('label_severity')
        ])
        if len(df_sample) < n_samples:
             restante = n_samples - len(df_sample)
             pool = df_temp.drop(df_sample.index)
             if not pool.empty:
                 extra = pool.sample(n=min(len(pool), restante), random_state=42)
                 df_sample = pd.concat([df_sample, extra])
    except:
        df_sample = df_temp.sample(n=n_samples, random_state=42)

    # ORDENAR por gravidade (0 -> 4)
    df_sample = df_sample.sort_values(by='label_severity')
    
    labels_sample = df_sample['label_severity']
    # Guardamos os índices originais para saber quem é quem
    indices_originais = df_sample.index 
    data_sample = df_sample.drop(columns=['label_severity'])
    
    # 2. Cálculo da Distância (Euclidiana, conforme Puccio et al. )
    dist_vector = pdist(data_sample, metric='euclidean')
    dist_matrix = squareform(dist_vector)
    
    # 3. Criar DataFrame Numérico (Para visualização tabular)
    # Usamos os índices originais e concatenamos com a gravidade para facilitar a leitura
    # Ex: "Index123 (Grau 0)"
    nomes_colunas = [f"Pct_{idx} (G{grau})" for idx, grau in zip(indices_originais, labels_sample)]
    
    df_matrix_numerica = pd.DataFrame(
        dist_matrix, 
        index=nomes_colunas, 
        columns=nomes_colunas
    )

    # 4. Plotagem com Anotação Numérica (annot=True)
    # Aumentamos muito o figsize para os números caberem
    plt.figure(figsize=(20, 18)) 
    
    sns.heatmap(
        df_matrix_numerica, 
        cmap='viridis_r', 
        annot=True,          # <--- ISSO ESCREVE OS NÚMEROS
        fmt=".1f",           # Formata para 1 casa decimal (ex: 2.4)
        annot_kws={"size": 8}, # Tamanho da fonte do número
        square=True,
        cbar_kws={"shrink": .5}
    )
    
    plt.title(f"Matriz Numérica Detalhada ({n_samples} Pacientes)\nOrdenado: Saudável (G0) -> Grave (G4)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.show()
    
    print("Abaixo, uma amostra dos valores numéricos (primeiros 5x5):")
    # Mostra apenas um pedacinho no console para não poluir
    print(df_matrix_numerica.iloc[:5, :5].round(2))
    
    return df_matrix_numerica

