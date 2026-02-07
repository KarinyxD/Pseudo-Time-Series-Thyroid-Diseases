from src import (
    preprocessing as pp,
    bootstrap as bt,
    trajectory as tj
)
import pandas as pd

# Pré-processamento dos dados
df_norm, df_real, severity_label = pp.preprocessing_pts()
pp.export_data_pp(df_norm, df_real, severity_label) # exportar dados preprocessamento
severity_label_serie = pd.Series(severity_label, index=df_norm.index)

# 2. Bootstrap (Pode mandar a Series ou o array, tanto faz)
amostras = bt.gerar_amostras_bootstrap(severity_label_serie, k=1500, T=30)

# 3. Processar as trajetórias (calcular matriz, gerar MST e ordenar)
trajetorias = tj.processar_todas_trajetorias(df_norm, severity_label_serie, amostras, qtd_plots=1)

# 4. Exportar as trajetorias para csv
tj.exportar_trajetorias(df_real, trajetorias, severity_label_serie, nome_arquivo="data/results/trajectories.csv")
