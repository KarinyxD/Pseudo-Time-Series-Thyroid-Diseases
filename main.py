import matriz_euclidiana as me
import preprocessamento as pp
import MST as mst
import bootstrap as bt
import pandas as pd


# Pré-processamento dos dados
df_norm, df_real, severity_label = pp.preprocessing_pts()
severity_label_serie = pd.Series(severity_label, index=df_norm.index)

# 2. Bootstrap (Pode mandar a Series ou o array, tanto faz)
amostras = bt.gerar_amostras_bootstrap(severity_label_serie, k=1500, T=30)

# 3. Processar as trajetórias (calcular matriz, gerar MST e ordenar)
trajetorias = bt.processar_todas_trajetorias(df_norm, severity_label_serie, amostras, qtd_plots=1)

bt.exportar_trajetorias_com_severidade(df_real, trajetorias, severity_label_serie, nome_arquivo="csv/trajetorias.csv")
