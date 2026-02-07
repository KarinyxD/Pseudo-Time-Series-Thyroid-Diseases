import matriz_euclidiana as me
import preprocessamento as pp
import MST as mst

# Pré-processamento dos dados
df_norm, df_real, severity_label = pp.preprocessing_pts()

# Gera matriz de distancia com os labels de severidade e retorna um Dataframe, n_samples define a dim. da matriz
# df_matrix = me.prepare_distance_dataframe(df_norm, severity_label, n_samples=None) # Matriz completa: n_samples=None


# df_features = seu dataframe JÁ filtrado com as colunas certas (TSH, T4, Idade...)
# labels = seus labels (0, 1, 2, 3)
# new.visualize_trajectory_check(df_norm, severity_label)

# pts_list = gen.generate_stratified_bootstrap_primary(df_norm, severity_label)
# df_csv = gen.save_pts_to_csv(pts_list, "csv/meus_dados_pts.csv")

# view.visualize_single_mst_path(df_norm, severity_label)




# 1. Converta para visualizar
# df_real vem do seu preprocessing_pts()
# pts_reais = gen.convert_trajectories_to_real_values(pts_list, df_real)

# 2. Salve um CSV "Legível para Humanos"
# gen.save_pts_to_csv(pts_reais, "csv/trajetorias_valores_reais.csv")



# df_matrix_small = me.prepare_distance_dataframe_amost(df_norm,severity_label,n_g0=45,n_g1=20,n_g2=20) # Amostragem com quantidades de cada grau
# Plota a matriz de distancia euclidiana com os valores numericos(parametro tem que ser um dataFrame)
# me.plot_numerical_matrix(df_matrix_small) # Pequena

# Plota a Matriz de distancia euclidiana sem os valores numericos(parametro tem que ser um dataFrame)
#me.plot_overview_heatmap(df_matrix) # Grande

# DataFrame da Matriz de Distância para CSV
# df_matrix_small.to_csv("csv/matriz_distancias.csv")

# Calcular MST
# grafo_mst = mst.mst(df_matrix)
# mst.plot_mst_graph_sample_mds(grafo_mst) # Plota o grafo da MST (considerando as distancias dos vértices)
# mst.plot_mds_full_mst(grafo_mst) # Plota o grafo geral da MST (sem considerar as distancias dos vértices)


