import MST as mst
import preprocessamento as pp
import distancia_geodesica as dg
import view as view
import data_longitudinal as dl

df_norm, df_real, labels = pp.preprocessing_pts()
mst_matrix, root_node, full_dist_matrix = mst.build_mst_and_root(df_norm, labels)
mst.plot_mst_pca(df_norm, mst_matrix, labels, root_node)
df_final = dg.calculate_pseudo_time(mst_matrix, root_node, df_real, labels, df_norm)
view.plot_trajectory(df_final)


# Gera o dataset final
df_simulated = dl.generate_longitudinal_dataset(mst_matrix, root_node, df_real, labels)
# 1. Salvar o DataFrame em um arquivo CSV local
# index=False evita salvar a coluna de índice numérico do Pandas (0, 1, 2...) que não precisamos
df_simulated.to_csv('simulacao_longitudinal_tireoide.csv', index=False)


# Visualizar o "Histórico" de um paciente virtual
print("\nExemplo de histórico do primeiro paciente virtual gerado:")
first_patient = df_simulated['sim_patient_id'].iloc[0]
print(df_simulated[df_simulated['sim_patient_id'] == first_patient][['time_step', 'TSH', 'real_class', 'age']])