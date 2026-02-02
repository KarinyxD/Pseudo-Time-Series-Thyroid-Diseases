# import MST as mst
import matriz_euclidiana as me
import preprocessamento as pp
import MST as mst
#import distancia_geodesica as dg
#import view as view
#import data_longitudinal as dl

# Pré-processamento dos dados
df_norm, df_real, severity_label = pp.preprocessing_pts()

# Gera matriz de distancia com os labels de severidade e retorna um Dataframe, n_samples define a dim. da matriz
# df_matrix = me.prepare_distance_dataframe(df_norm, severity_label, n_samples=50)

# Plota a matriz de distancia euclidiana com os valores numericos(parametro tem que ser um dataFrame)
# df_matrix_small = me.prepare_distance_dataframe(df_norm, severity_label, n_samples=150)
df_matrix_small = me.prepare_distance_dataframe_amost(df_norm,severity_label,n_g0=45,n_g1=20,n_g2=20) # Amostragem com quantidades de cada grau
# me.plot_numerical_matrix(df_matrix_small) # Pequena

# Plota a Matriz de distancia euclidiana sem os valores numericos(parametro tem que ser um dataFrame)
#me.plot_overview_heatmap(df_matrix) # Grande

# Calcular MST
grafo_mst = mst.compute_mst(df_matrix_small)
mst.plot_mst_graph_sample_mds(grafo_mst)
# mst.plot_mst_graph(grafo_mst) # Plotar o grafo MST

# mst_matrix, root_node, full_dist_matrix = mst.build_mst_and_root(df_norm, labels)
# mst.plot_mst_pca(df_norm, mst_matrix, labels, root_node)
# df_final = dg.calculate_pseudo_time(mst_matrix, root_node, df_real, labels, df_norm)
# view.plot_trajectory(df_final)


# # Gera o dataset final
# df_simulated = dl.generate_longitudinal_dataset(mst_matrix, root_node, df_real, labels)
# # 1. Salvar o DataFrame em um arquivo CSV local
# # index=False evita salvar a coluna de índice numérico do Pandas (0, 1, 2...) que não precisamos
# df_simulated.to_csv('simulacao_longitudinal_tireoide.csv', index=False)


# # Visualizar o "Histórico" de um paciente virtual
# print("\nExemplo de histórico do primeiro paciente virtual gerado:")
# first_patient = df_simulated['sim_patient_id'].iloc[0]
# print(df_simulated[df_simulated['sim_patient_id'] == first_patient][['time_step', 'TSH', 'real_class', 'age']])