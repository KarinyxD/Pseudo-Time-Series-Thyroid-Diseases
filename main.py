import matriz_euclidiana as me
import preprocessamento as pp
import MST as mst

# Pré-processamento dos dados
df_norm, df_real, severity_label = pp.preprocessing_pts()

# Gera matriz de distancia com os labels de severidade e retorna um Dataframe, n_samples define a dim. da matriz
# df_matrix = me.prepare_distance_dataframe(df_norm, severity_label, n_samples=50)

# Plota a matriz de distancia euclidiana com os valores numericos(parametro tem que ser um dataFrame)
df_matrix_small = me.prepare_distance_dataframe_amost(df_norm,severity_label,n_g0=45,n_g1=20,n_g2=20) # Amostragem com quantidades de cada grau
# me.plot_numerical_matrix(df_matrix_small) # Pequena

# Plota a Matriz de distancia euclidiana sem os valores numericos(parametro tem que ser um dataFrame)
#me.plot_overview_heatmap(df_matrix) # Grande

# Calcular MST
grafo_mst = mst.compute_mst(df_matrix_small)
mst.plot_mst_graph_sample_mds(grafo_mst) # Plota o grafo da MST (considerando as distancias dos vértices)
# mst.plot_mst_graph(grafo_mst) # Plota o grafo geral da MST (sem considerar as distancias dos vértices)


