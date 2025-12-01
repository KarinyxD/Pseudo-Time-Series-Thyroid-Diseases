from scipy.sparse.csgraph import shortest_path
import preprocessamento as pp

def calculate_pseudo_time(mst_matrix, root_node_idx, df_real, labels, df_normalized):
    print("Calculando distâncias geodésicas (Caminho mais curto na árvore)...")
    
    # Esta função calcula a distância do 'indices' (raiz) para TODOS os outros nós
    # directed=False é importante porque o grafo não tem "mão única"
    dist_matrix_geodesic = shortest_path(csgraph=mst_matrix, directed=False, indices=root_node_idx)
    
    # Adicionamos os resultados ao DataFrame "Real" (o que tem os valores originais)
    # Assim podemos ver: TSH=100 corresponde a qual Pseudo-Time?
    df_results = df_real.copy()
    df_results['pseudo_time'] = dist_matrix_geodesic
    df_results['label'] = labels.values # Trazemos as classes originais (-, G, F...)
    df_results['target_clean'] = df_results['label'].map({v: k for k, v in pp.class_mapping_details.items()})
    
    # Também guardamos o TSH normalizado para plots comparativos se precisar
    df_results['TSH_norm'] = df_normalized['TSH'].values 
    
    print("Cálculo concluído.")
    print("Exemplo dos primeiros 5 pacientes (ordenados por tempo):")
    print(df_results[['pseudo_time', 'target_clean', 'TSH', 'TT4']].sort_values('pseudo_time').head())
    
    return df_results

# --- EXECUÇÃO DO PASSO 3 ---
# df_real é o df_imputed que retornamos na função preprocessing
# df_final = calculate_pseudo_time(mst_matrix, root_node, df_real, labels, df_norm)